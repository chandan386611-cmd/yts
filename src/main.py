"""
Automated YouTube Shorts Pipeline Orchestrator.

Coordinates script generation, voiceover duration gating, visual generation,
subtitles, video rendering, and conditional YouTube API publishing.

Enforces publishing safeguards:
  - --dry-run renders locally without calling YouTube API or modifying history.json.
  - Normal mode updates history.json ONLY after a verified YouTube video ID is returned.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.caption_generator import CaptionGenerator
from src.config import config
from src.script_generator import ScriptGenerator, ShortScript
from src.video_editor import VideoEditor
from src.visual_generator import VisualGenerator
from src.voice_generator import VoiceGenerator
from src.youtube_uploader import YouTubeUploader


def setup_structured_logging() -> logging.Logger:
    """Configures structured console logging with timestamp and component tags."""
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
    return logging.getLogger("pipeline")


logger = setup_structured_logging()


def record_successful_upload(
    script: ShortScript,
    video_id: str,
    duration_seconds: float,
) -> None:
    """
    Appends the successfully published video record to data/history.json.
    MUST ONLY be called after YouTube uploader confirms a valid video ID.
    """
    history_file = config.HISTORY_FILE
    now_str = datetime.now(timezone.utc).isoformat()

    entry = {
        "topic": script.topic,
        "title": script.title,
        "video_id": video_id,
        "url": f"https://www.youtube.com/shorts/{video_id}",
        "published_at": now_str,
        "duration_seconds": round(duration_seconds, 2),
        "total_words": script.total_words,
        "scenes_count": len(script.scenes),
    }

    try:
        history_data: dict[str, Any] = {"used_topics": []}
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)

        used_topics: list[Any] = history_data.setdefault("used_topics", [])
        used_topics.append(entry)

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)

        logger.info("Successfully updated %s with new video ID: %s", history_file, video_id)
    except Exception as e:
        logger.critical("Failed to write to history.json after upload: %s", e)
        raise


def clean_temp_directory() -> None:
    """Removes intermediate temp assets while preserving directory structure."""
    if config.TEMP_DIR.exists():
        logger.info("Cleaning up temporary processing files in %s...", config.TEMP_DIR)
        for item in config.TEMP_DIR.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                logger.debug("Failed to remove temporary item %s: %s", item, e)


def run_pipeline(
    dry_run: bool = False,
    custom_topic: str | None = None,
    keep_temp: bool = False,
    force_edge_tts: bool = False,
    custom_script: ShortScript | None = None,
) -> int:
    """
    Executes the automated Shorts generation and publishing pipeline.

    Returns:
        0 on success, non-zero on failure.
    """
    logger.info("=================================================================")
    logger.info("  AUTONOMOUS YOUTUBE SHORTS PIPELINE INITIATED")
    logger.info("  Mode: %s", "DRY-RUN (Local Only)" if dry_run else "PRODUCTION (Live Publish)")
    logger.info("  Niche: %s", config.NICHE_TOPIC)
    logger.info("=================================================================")

    if force_edge_tts:
        config.USE_EDGE_TTS_FALLBACK = True

    # 1. Initialize environment directories and placeholder assets
    config.init_directories()
    if not config.BG_MUSIC_PATH.exists() or not config.WHOOSH_PATH.exists():
        from scripts.generate_default_assets import main as gen_assets
        gen_assets()

    try:
        # 2. Topic deduplication & Structured Script Generation
        logger.info("--- STAGE 1: SCRIPT & VISUAL PROMPT GENERATION ---")
        script_gen = ScriptGenerator()
        if custom_script:
            initial_script = custom_script
            logger.info("Using user-provided custom script: '%s'", initial_script.topic)
        else:
            initial_script = script_gen.generate_script(custom_topic=custom_topic)
        logger.info("Topic: '%s'", initial_script.topic)
        logger.info("Title: '%s'", initial_script.title)
        logger.info("Scenes: %d | Total Words: %d", len(initial_script.scenes), initial_script.total_words)

        # 3. Voiceover Generation & Audio Duration Gate (40.0s - 45.0s)
        logger.info("--- STAGE 2: VOICEOVER SYNTHESIS & DURATION GATE ---")
        voice_gen = VoiceGenerator()
        final_script, scene_audios, merged_audio, total_duration = voice_gen.generate_and_gate_audio(
            script=initial_script,
            script_generator=script_gen,
            max_regeneration_attempts=3,
        )

        logger.info(
            "Gate verified. Total Spoken Duration: %.2fs across %d scenes",
            total_duration,
            len(scene_audios),
        )

        # 4. Hybrid Visual Asset Generation
        logger.info("--- STAGE 3: HYBRID VISUAL ASSET GENERATION (REAL STOCK VIDEO + STILL FALLBACK) ---")
        visual_gen = VisualGenerator()
        scene_visuals = visual_gen.generate_all_visuals(final_script)
        video_count = sum(1 for v in scene_visuals if v.suffix.lower() == ".mp4")
        still_count = len(scene_visuals) - video_count
        logger.info(
            "Visual assets ready (Total Scenes: %d | Real Video Clips: %d | Animated Stills: %d)",
            len(scene_visuals),
            video_count,
            still_count,
        )

        # 5. Caption Generation (.ass with dynamic pop-in animations)
        logger.info("--- STAGE 4: DYNAMIC CENTER-SCREEN CAPTION GENERATION ---")
        caption_gen = CaptionGenerator()
        ass_subtitles = caption_gen.generate_subtitles(
            merged_audio_path=merged_audio,
            script=final_script,
            scene_audios=scene_audios,
            total_duration=total_duration,
        )

        # 6. FFmpeg Assembly, Ken Burns Animation, Audio Mixing & Subtitle Burn
        logger.info("--- STAGE 5: FFMPEG ASSEMBLY & SUBTITLE BURN-IN ---")
        editor = VideoEditor()
        final_video_path = editor.build_video(
            scene_visuals=scene_visuals,
            scene_audios=scene_audios,
            merged_audio_path=merged_audio,
            subtitles_ass_path=ass_subtitles,
            total_duration=total_duration,
        )

        logger.info("Rendered video file: %s", final_video_path)

        # 7. Publishing & Safeguards
        if dry_run:
            logger.info("=================================================================")
            logger.info("  DRY-RUN COMPLETE: Video generated successfully.")
            logger.info("  Location: %s", final_video_path.resolve())
            logger.info("  YouTube API was NOT called. history.json was NOT modified.")
            logger.info("=================================================================")
            return 0

        # Production Mode: Upload to YouTube
        logger.info("--- STAGE 6: OFFICIAL YOUTUBE DATA API V3 PUBLISHING ---")
        uploader = YouTubeUploader()
        video_id = uploader.upload_short(video_path=final_video_path, script=final_script)

        # 8. Update history.json ONLY AFTER verified upload
        record_successful_upload(
            script=final_script,
            video_id=video_id,
            duration_seconds=total_duration,
        )

        # 9. Clean up temporary directory
        if not keep_temp:
            clean_temp_directory()

        logger.info("=================================================================")
        logger.info("  PIPELINE EXECUTED SUCCESSFULLY")
        logger.info("  YouTube Short Live: https://www.youtube.com/shorts/%s", video_id)
        logger.info("=================================================================")
        return 0

    except Exception as exc:
        logger.critical("PIPELINE EXECUTION FAILED: %s", exc)
        traceback.print_exc()
        logger.critical("Publishing aborted. history.json has NOT been modified.")
        return 1


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="Autonomous YouTube Shorts Generation and Publishing Pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and render video locally without publishing to YouTube or updating history.json.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Custom topic prompt to override automated topic selection.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve temporary audio, clip, and subtitle files for inspection.",
    )
    parser.add_argument(
        "--force-edge-tts",
        action="store_true",
        help="Force Edge-TTS synthesis instead of ElevenLabs.",
    )

    args = parser.parse_args()
    exit_code = run_pipeline(
        dry_run=args.dry_run,
        custom_topic=args.topic,
        keep_temp=args.keep_temp,
        force_edge_tts=args.force_edge_tts,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
