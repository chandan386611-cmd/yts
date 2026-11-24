"""
Dynamic Center-Screen Caption Generator.

Transcribes merged voiceover audio using faster-whisper (base.en, int8 on CPU)
to extract exact word-level timestamps, formats dynamic 2-3 word chunks,
and renders styled .ass (Advanced SubStation Alpha) subtitles with high-contrast
pop-in animations optimized for vertical YouTube Shorts.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import config
from src.script_generator import ShortScript

logger = logging.getLogger("caption_generator")


@dataclass
class WordTimestamp:
    """Individual word timing."""
    word: str
    start: float
    end: float


def format_ass_timestamp(seconds: float) -> str:
    """Convert float seconds to ASS format H:MM:SS.cs (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - math.floor(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def generate_ass_header() -> str:
    """Generate ASS script header and custom styling for 1080x1920 vertical canvas."""
    # ASS color format is &HAABBGGRR (Alpha, Blue, Green, Red)
    # White: &H00FFFFFF
    # Bright Yellow: &H0000FFFF (BB=00, GG=FF, RR=FF)
    # Outline Black: &H00000000
    # Shadow Deep Black: &HA0000000
    # Alignment: 2 (Bottom-Center)
    # MarginV: 380 (Sweet Spot: above bottom YouTube UI, below center face area)
    # MarginL: 60, MarginR: 180 (Safely clears YouTube right-side like/comment buttons)
    return """[Script Info]
Title: Dynamic YouTube Shorts Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ShortsDefault,Arial Black,82,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,2,0,1,8,4,2,60,180,380,1
Style: ShortsHighlight,Arial Black,84,&H0000FFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,105,105,2,0,1,8,4,2,60,180,380,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class CaptionGenerator:
    """Extracts word timestamps and renders styled .ass subtitle files."""

    def __init__(self) -> None:
        self.output_ass_path = config.TEMP_DIR / "subtitles.ass"

    def _transcribe_with_faster_whisper(self, audio_path: Path) -> list[WordTimestamp]:
        """Run faster-whisper on merged voiceover audio on CPU using int8 quantization."""
        logger.info("Transcribing audio with faster-whisper (base.en, int8)...")
        from faster_whisper import WhisperModel

        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            language="en",
        )

        words_list: list[WordTimestamp] = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    cleaned_word = w.word.strip()
                    if cleaned_word:
                        words_list.append(
                            WordTimestamp(word=cleaned_word, start=w.start, end=w.end)
                        )

        logger.info("faster-whisper extracted %d timed words.", len(words_list))
        return words_list

    def _fallback_word_timestamps(
        self,
        script: ShortScript,
        scene_audios: list[tuple[Path, float]],
        total_duration: float,
    ) -> list[WordTimestamp]:
        """
        Calculates proportional word timestamps from exact scene audio durations
        when faster-whisper is unavailable.
        """
        logger.info("Synthesizing word timestamps from scene audio durations...")
        words_list: list[WordTimestamp] = []
        current_time = 0.0

        for idx, scene in enumerate(script.scenes):
            scene_duration = scene_audios[idx][1] if idx < len(scene_audios) else 4.0
            # Use Romanized display_text for subtitles if present, otherwise spoken narration
            sub_text = scene.display_text if getattr(scene, "display_text", None) else scene.narration
            words = sub_text.split()
            if not words:
                current_time += scene_duration
                continue

            time_per_word = scene_duration / len(words)
            for w in words:
                w_start = current_time
                w_end = current_time + time_per_word
                words_list.append(
                    WordTimestamp(
                        word=w.strip(),
                        start=round(w_start, 3),
                        end=round(w_end, 3),
                    )
                )
                current_time += time_per_word

        return words_list

    def build_ass_subtitles(
        self,
        words_list: list[WordTimestamp],
        output_file: Path | None = None,
    ) -> Path:
        """
        Groups words into ultra-punchy 1-2 word chunks and writes dynamic pop-in ASS events
        repositioned to the lower-third reading sweet spot.
        """
        dest_path = output_file or self.output_ass_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [generate_ass_header()]

        total_words = len(words_list)
        chunk_idx = 0

        i = 0
        while i < total_words:
            # High-retention chunking: 1-2 words per visual pop
            # If word is 8+ chars (long word) or at sentence end, keep as single word chunk
            curr_word = words_list[i].word
            remaining = total_words - i
            if remaining == 1 or len(curr_word) >= 8:
                step = 1
            else:
                step = 2

            end_i = min(i + step, total_words)
            chunk = words_list[i:end_i]
            if not chunk:
                break

            start_time = chunk[0].start
            end_time = max(chunk[-1].end, start_time + 0.32)

            start_ass = format_ass_timestamp(start_time)
            end_ass = format_ass_timestamp(end_time)

            # Format words: highlight first word in electric neon yellow, second in crisp white
            formatted_words = []
            for j, w in enumerate(chunk):
                cleaned = re.sub(r"[^\w\s\'\-]", "", w.word).upper()
                if not cleaned:
                    cleaned = w.word.upper()

                if j == 0:
                    formatted_words.append(r"{\c&H0000FFFF&}" + cleaned + r"{\c&H00FFFFFF&}")
                else:
                    formatted_words.append(cleaned)

            text_body = " ".join(formatted_words)
            # High-retention pop-in scale bounce animation: 115% scale dropping to 100% in 90ms
            animation_tag = r"{\t(0,50,\fscx115\fscy115)\t(50,110,\fscx100\fscy100)}"
            event_line = f"Dialogue: 0,{start_ass},{end_ass},ShortsDefault,,0,0,0,,{animation_tag}{text_body}\n"
            lines.append(event_line)

            chunk_idx += 1
            i = end_i

        with open(dest_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.info("Successfully written ASS subtitle file with %d dialogue chunks to %s", chunk_idx, dest_path)
        return dest_path

    def generate_subtitles(
        self,
        merged_audio_path: Path,
        script: ShortScript,
        scene_audios: list[tuple[Path, float]],
        total_duration: float,
    ) -> Path:
        """
        Coordinates transcription and subtitle file generation.
        For Hinglish, uses exact scene display_text for 100% flawless phonetic spelling.
        For English, runs faster-whisper with proportional timing fallback.
        """
        words: list[WordTimestamp] = []
        is_hinglish = getattr(config, "LANGUAGE", "hinglish").lower() in ("hinglish", "hindi")

        if is_hinglish:
            logger.info("Language is Hinglish/Hindi: using script display_text for perfect Romanized on-screen subtitles.")
            words = self._fallback_word_timestamps(script, scene_audios, total_duration)
        else:
            try:
                words = self._transcribe_with_faster_whisper(merged_audio_path)
            except Exception as e:
                logger.warning("faster-whisper transcription unavailable (%s). Using timing fallback.", e)

        if not words:
            words = self._fallback_word_timestamps(script, scene_audios, total_duration)

        return self.build_ass_subtitles(words)
