"""
Publishes the already-rendered final_short.mp4 to YouTube Shorts.
"""

from pathlib import Path
from src.config import config
from src.script_generator import ShortScript, ScenePrompt
from src.youtube_uploader import YouTubeUploader
from src.main import record_successful_upload
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("publish_existing")

def main():
    video_path = config.OUTPUT_DIR / "final_short.mp4"
    if not video_path.exists():
        logger.error("Video file %s not found!", video_path)
        sys.exit(1)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    logger.info("Found final short at %s (%.2f MB)", video_path, file_size_mb)

    script = ShortScript(
        topic="Neural Dust Micro-Implants",
        title="Your thoughts are no longer private! #NeuroTech #Future #Shorts",
        description="Kya bhavishya mein humare vichaar screen par live dikhenge? Neural Dust technology ka khaufnaak sach! #Science #Technology #NeuroTech #Future #AI #Shorts",
        tags=["science", "future", "neurotech", "technology", "ai", "shorts", "futuretech"],
        scenes=[
            ScenePrompt(
                scene_id=0,
                narration="Imagine kijiye aapke vichaar ab kisi aur ke screen par live dikh rahe hain...",
                visual_prompt="Futuristic neural network brain scan",
                estimated_duration=4.5,
            )
        ]
    )

    uploader = YouTubeUploader()
    logger.info("Starting upload to YouTube channel Zorvian...")
    video_id = uploader.upload_short(video_path, script)
    logger.info("Upload completed! Video ID: %s", video_id)

    record_successful_upload(script, video_id, duration_seconds=43.97)
    logger.info("Published Live Short URL: https://www.youtube.com/shorts/%s", video_id)
    print(f"\nSUCCESS_URL: https://www.youtube.com/shorts/{video_id}")

if __name__ == "__main__":
    main()
