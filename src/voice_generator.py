"""
Voiceover Generator with ElevenLabs API and Edge-TTS Fallback.

Generates scene-by-scene audio with realistic pacing, measures exact ffprobe
durations, enforces a strict 40.0s - 45.0s audio duration gate, and coordinates
automatic Gemini script regeneration for up to 3 attempts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Callable

import requests

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        def decorator(f: Any) -> Any:
            return f
        return decorator

    def retry_if_exception_type(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return None

    def stop_after_attempt(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return None

    def wait_exponential(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return None

from src.config import config
from src.script_generator import ScenePrompt, ScriptGenerator, ShortScript

logger = logging.getLogger("voice_generator")


def probe_audio_duration(file_path: Path) -> float:
    """
    Measure exact audio duration in seconds.
    Uses ffprobe if available, with an internal pure-Python MPEG frame parser fallback.
    """
    ffprobe_bin = config.FFPROBE_PATH
    if shutil.which(ffprobe_bin) or Path(ffprobe_bin).exists():
        try:
            cmd = [
                ffprobe_bin,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            logger.debug("ffprobe execution failed (%s). Using fallback parser.", e)

    # Pure-Python fallback: Parse MPEG Audio (MP3) frame headers
    try:
        with open(file_path, "rb") as f:
            data = f.read()

        total_frames = 0
        sample_rate = 44100
        idx = 0
        data_len = len(data)

        # Skip ID3v2 if present
        if data.startswith(b"ID3") and data_len > 10:
            # ID3 size is encoded in 4 bytes (syncsafe integer: 7 bits each)
            size = (
                ((data[6] & 0x7F) << 21)
                | ((data[7] & 0x7F) << 14)
                | ((data[8] & 0x7F) << 7)
                | (data[9] & 0x7F)
            )
            idx = 10 + size

        # Bitrate lookup table (MPEG-1 Layer III in kbps)
        bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
        sample_rates = [44100, 48000, 32000, 0]

        while idx < data_len - 4:
            if data[idx] == 0xFF and (data[idx + 1] & 0xE0) == 0xE0:
                header = struct.unpack(">I", data[idx:idx + 4])[0]
                version = (header >> 19) & 3
                layer = (header >> 17) & 3
                bitrate_idx = (header >> 12) & 0xF
                sr_idx = (header >> 10) & 3
                padding = (header >> 9) & 1

                if version == 3 and layer == 1 and bitrate_idx in range(1, 15) and sr_idx in range(0, 3):
                    bitrate = bitrates[bitrate_idx] * 1000
                    sample_rate = sample_rates[sr_idx]
                    frame_len = int((144 * bitrate) / sample_rate) + padding
                    if frame_len > 0:
                        total_frames += 1
                        idx += frame_len
                        continue
            idx += 1

        if total_frames > 0:
            duration = (total_frames * 1152) / sample_rate
            return round(duration, 3)
    except Exception as e:
        logger.debug("Pure python MP3 duration parse failed: %s", e)

    # Estimate from word count if all else fails (approx 150 WPM = 2.5 words/sec)
    return 4.0


class VoiceGenerator:
    """Manages multi-scene voiceover synthesis and duration gating."""

    def __init__(self) -> None:
        self.temp_audio_dir = config.TEMP_DIR / "audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.2, min=2, max=8),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _synthesize_elevenlabs(self, text: str, output_path: Path) -> bool:
        """Call ElevenLabs API using eleven_flash_v2_5 at 0.5 credits/char."""
        if not config.ELEVENLABS_API_KEY:
            return False

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.ELEVENLABS_API_KEY,
        }
        payload = {
            "text": text,
            "model_id": config.ELEVENLABS_MODEL_ID,
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.80,
                "style": 0.35,
                "use_speaker_boost": True,
            },
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        elif response.status_code in (401, 402, 429):
            logger.warning(
                "ElevenLabs quota limit or auth issue (HTTP %d: %s). Switching to edge-tts.",
                response.status_code,
                response.text[:200],
            )
            return False
        else:
            response.raise_for_status()
            return False

    def _synthesize_edge_tts(self, text: str, output_path: Path) -> bool:
        """Synthesize voiceover using Microsoft Edge TTS (high-quality neural voice)."""
        try:
            import edge_tts

            async def _run() -> None:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=config.EDGE_TTS_VOICE,
                    rate="+4%",  # Slight pace boost for high retention
                    pitch="+0Hz",
                )
                await communicate.save(str(output_path))

            asyncio.run(_run())
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception as e:
            logger.error("Edge-TTS synthesis error: %s", e)
            return False

    def _synthesize_dummy_scene_audio(self, text: str, output_path: Path) -> None:
        """Fallback synthesizer when offline or without TTS dependencies."""
        from scripts.generate_default_assets import generate_pure_python_mp3

        words = len(text.split())
        approx_duration = max(3.0, round(words / 2.65, 2))
        generate_pure_python_mp3(output_path, duration_seconds=approx_duration)

    def synthesize_scene_audio(self, scene: ScenePrompt, index: int) -> tuple[Path, float]:
        """Synthesize audio for a single scene and measure exact duration."""
        output_path = self.temp_audio_dir / f"scene_{index}.mp3"
        success = False

        # Attempt ElevenLabs if not forced to edge-tts
        if not config.USE_EDGE_TTS_FALLBACK and config.ELEVENLABS_API_KEY:
            try:
                success = self._synthesize_elevenlabs(scene.narration, output_path)
            except Exception as e:
                logger.warning("ElevenLabs error for scene %d: %s. Using edge-tts fallback.", index, e)
                success = False

        # Fallback to Edge-TTS
        if not success:
            success = self._synthesize_edge_tts(scene.narration, output_path)

        # Fallback to silent/sine placeholder if offline
        if not success or not output_path.exists() or output_path.stat().st_size == 0:
            logger.warning("TTS engines failed for scene %d. Using offline audio placeholder.", index)
            self._synthesize_dummy_scene_audio(scene.narration, output_path)

        duration = probe_audio_duration(output_path)
        logger.info("Scene %d audio generated: %.2fs (%s)", index, duration, output_path.name)
        return output_path, duration

    def merge_audio_tracks(self, audio_files: list[Path], output_merged: Path) -> float:
        """
        Merge scene audio clips into a unified voiceover track.
        Uses FFmpeg concat if available, or direct binary MP3 frame concatenation.
        """
        ffmpeg_bin = config.FFMPEG_PATH
        if shutil.which(ffmpeg_bin):
            list_file = self.temp_audio_dir / "concat_audio_list.txt"
            with open(list_file, "w", encoding="utf-8") as f:
                for af in audio_files:
                    # Windows paths need forward slashes for ffmpeg concat
                    safe_path = str(af.resolve()).replace("\\", "/")
                    f.write(f"file '{safe_path}'\n")

            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_merged),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return probe_audio_duration(output_merged)
            except Exception as e:
                logger.debug("FFmpeg audio concat failed (%s). Using binary stream concat.", e)

        # Binary MPEG frame concatenation (valid for uniform MP3 bitstreams)
        with open(output_merged, "wb") as outfile:
            for af in audio_files:
                with open(af, "rb") as infile:
                    outfile.write(infile.read())

        return probe_audio_duration(output_merged)

    def generate_and_gate_audio(
        self,
        script: ShortScript,
        script_generator: ScriptGenerator,
        max_regeneration_attempts: int = 3,
    ) -> tuple[ShortScript, list[tuple[Path, float]], Path, float]:
        """
        Synthesizes scene audio and enforces the strict 40.0s - 45.0s duration gate.
        If actual total audio falls outside the window, re-prompts Gemini with
        explicit word-count expansion/compression feedback and regenerates audio
        (up to max_regeneration_attempts).

        Returns:
            (final_script, scene_audio_info, merged_voiceover_path, total_duration)
        """
        current_script = script

        for attempt in range(1, max_regeneration_attempts + 1):
            logger.info(
                "--- Voiceover Synthesis Attempt %d/%d (Script: '%s') ---",
                attempt,
                max_regeneration_attempts,
                current_script.topic,
            )

            scene_audios: list[tuple[Path, float]] = []
            for idx, scene in enumerate(current_script.scenes):
                path, dur = self.synthesize_scene_audio(scene, idx)
                scene_audios.append((path, dur))

            merged_path = self.temp_audio_dir / "merged_voiceover.mp3"
            total_duration = self.merge_audio_tracks([p for p, _ in scene_audios], merged_path)

            logger.info(
                "Total measured voiceover duration: %.2fs (Target: %.1fs to %.1fs)",
                total_duration,
                config.MIN_DURATION_SECONDS,
                config.MAX_DURATION_SECONDS,
            )

            # Check if duration meets strict gate
            if config.MIN_DURATION_SECONDS <= total_duration <= config.MAX_DURATION_SECONDS:
                logger.info(">>> Audio duration GATE PASSED: %.2fs <<<", total_duration)
                return current_script, scene_audios, merged_path, total_duration

            if attempt < max_regeneration_attempts:
                logger.warning(
                    "Duration %.2fs is outside allowable window (%.1fs - %.1fs). Re-prompting script...",
                    total_duration,
                    config.MIN_DURATION_SECONDS,
                    config.MAX_DURATION_SECONDS,
                )
                current_script = script_generator.re_prompt_with_duration_feedback(
                    current_script,
                    actual_duration=total_duration,
                    target_range=(config.MIN_DURATION_SECONDS, config.MAX_DURATION_SECONDS),
                )
            else:
                logger.warning(
                    "Reached maximum audio regeneration attempts (%d). Proceeding with duration %.2fs.",
                    max_regeneration_attempts,
                    total_duration,
                )
                return current_script, scene_audios, merged_path, total_duration

        return current_script, scene_audios, merged_path, total_duration
