"""
Central Configuration & Pacing Constants for YouTube Shorts Pipeline.

Manages environment variable validation, art-direction style locks,
pacing/duration constraints, file paths, and hard cost spending guards.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Lightweight pure-Python .env parser fallback
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        try:
            with open(_env_file, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip()
                    # Strip wrapping quotes if any
                    if len(_v) >= 2 and (_v[0] == _v[-1] and _v[0] in ('"', "'")):
                        _v = _v[1:-1]
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
        except Exception:
            pass


logger = logging.getLogger("config")


def _find_ffmpeg() -> str:
    env_p = os.getenv("FFMPEG_PATH")
    if env_p and (shutil.which(env_p) or Path(env_p).exists()):
        return env_p
    which_p = shutil.which("ffmpeg")
    if which_p:
        return which_p
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_dir.exists():
            for candidate in winget_dir.rglob("ffmpeg.exe"):
                if candidate.is_file():
                    return str(candidate)
    return "ffmpeg"


def _find_ffprobe() -> str:
    env_p = os.getenv("FFPROBE_PATH")
    if env_p and (shutil.which(env_p) or Path(env_p).exists()):
        return env_p
    which_p = shutil.which("ffprobe")
    if which_p:
        return which_p
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_dir.exists():
            for candidate in winget_dir.rglob("ffprobe.exe"):
                if candidate.is_file():
                    return str(candidate)
    return "ffprobe"


class PipelineConfig(BaseModel):
    """Immutable central pipeline configuration and constraints."""

    # Project Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    ASSETS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "assets")
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "output")
    TEMP_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "temp")

    HISTORY_FILE: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "history.json")
    SPEND_TRACKER_FILE: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "spend_tracker.json"
    )

    BG_MUSIC_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "assets" / "bg_music.mp3"
    )
    WHOOSH_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "assets" / "whoosh.mp3"
    )

    # Niche & Art-Direction Style Locks
    NICHE_TOPIC: str = Field(
        default_factory=lambda: os.getenv("NICHE_TOPIC", "Dark Psychology & Human Behavior Facts").strip()
    )
    LANGUAGE: str = Field(
        default_factory=lambda: os.getenv("LANGUAGE", "hinglish")
    )
    ART_STYLE_SUFFIX: str = (
        "35mm cinematic film grain, dramatic chiaroscuro lighting, hyper-detailed, "
        "moody atmosphere, vertical 9:16 composition, no text, no watermarks"
    )

    # Pacing and Video Duration Constraints (Strict 40.0s - 45.0s window)
    MIN_DURATION_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("MIN_DURATION_SECONDS", "40.0"))
    )
    MAX_DURATION_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("MAX_DURATION_SECONDS", "45.0"))
    )
    TARGET_MIN_WORDS: int = Field(
        default_factory=lambda: int(os.getenv("TARGET_MIN_WORDS", "105"))
    )
    TARGET_MAX_WORDS: int = Field(
        default_factory=lambda: int(os.getenv("TARGET_MAX_WORDS", "120"))
    )
    MIN_SCENES: int = 7
    MAX_SCENES: int = 10

    # Video Output Specifications
    VIDEO_WIDTH: int = 1080
    VIDEO_HEIGHT: int = 1920
    VIDEO_FPS: int = 30
    BG_MUSIC_VOLUME_DB: float = Field(
        default_factory=lambda: float(os.getenv("BG_MUSIC_VOLUME_DB", "-22.0"))
    )

    # Script Generation (ChatGPT OpenAI & Google Gemini)
    SCRIPT_PROVIDER: str = Field(
        default_factory=lambda: os.getenv("SCRIPT_PROVIDER", "openai" if os.getenv("OPENAI_API_KEY") else "gemini").lower()
    )
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))

    # Voiceover Generation (Cartesia Sonic + ElevenLabs + Edge-TTS Fallback)
    CARTESIA_API_KEY: str = Field(default_factory=lambda: os.getenv("CARTESIA_API_KEY", ""))
    CARTESIA_VOICE_ID: str = Field(
        default_factory=lambda: os.getenv("CARTESIA_VOICE_ID", "7e8cb11d-37af-476b-ab8f-25da99b18644")
    )
    CARTESIA_MODEL: str = Field(
        default_factory=lambda: os.getenv("CARTESIA_MODEL", "sonic-latest")
    )
    ELEVENLABS_API_KEY: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    ELEVENLABS_VOICE_ID: str = Field(
        default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
    )
    ELEVENLABS_MODEL_ID: str = Field(
        default_factory=lambda: os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    )
    USE_EDGE_TTS_FALLBACK: bool = Field(
        default_factory=lambda: os.getenv("USE_EDGE_TTS_FALLBACK", "true").lower() in ("true", "1", "yes")
    )
    EDGE_TTS_VOICE: str = Field(
        default_factory=lambda: os.getenv(
            "EDGE_TTS_VOICE",
            "hi-IN-MadhurNeural" if os.getenv("LANGUAGE", "hinglish").lower() in ("hinglish", "hindi") else "en-US-ChristopherNeural"
        )
    )
    VOICE_RATE: str = Field(
        default_factory=lambda: os.getenv("VOICE_RATE", "+14%")
    )

    # Visual Generation - Scene 0 Fal.ai Kling Video
    ENABLE_FAL_HOOK: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_FAL_HOOK", "false").lower() in ("true", "1", "yes")
    )
    FAL_KEY: str = Field(default_factory=lambda: os.getenv("FAL_KEY", ""))
    FAL_DAILY_BUDGET_USD: float = Field(
        default_factory=lambda: float(os.getenv("FAL_DAILY_BUDGET_USD", "1.00"))
    )
    FAL_MONTHLY_BUDGET_USD: float = Field(
        default_factory=lambda: float(os.getenv("FAL_MONTHLY_BUDGET_USD", "25.00"))
    )
    FAL_ESTIMATED_COST_PER_CALL: float = 0.15  # Kling turbo 5s video approx cost

    # Visual Generation - Stock Video & AI Engines
    COVERR_API_KEY: str = Field(default_factory=lambda: os.getenv("COVERR_API_KEY", ""))
    PEXELS_API_KEY: str = Field(default_factory=lambda: os.getenv("PEXELS_API_KEY", ""))
    CLOUDFLARE_ACCOUNT_ID: str = Field(default_factory=lambda: os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    CLOUDFLARE_API_TOKEN: str = Field(default_factory=lambda: os.getenv("CLOUDFLARE_API_TOKEN", ""))
    CLOUDFLARE_FLUX_MODEL: str = Field(
        default_factory=lambda: os.getenv("CLOUDFLARE_FLUX_MODEL", "@cf/black-forest-labs/flux-1-schnell")
    )

    # YouTube Data API v3 & Publishing
    YOUTUBE_CLIENT_ID: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_ID", ""))
    YOUTUBE_CLIENT_SECRET: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET", ""))
    YOUTUBE_REFRESH_TOKEN: str = Field(default_factory=lambda: os.getenv("YOUTUBE_REFRESH_TOKEN", ""))
    YOUTUBE_PRIVACY_STATUS: str = Field(
        default_factory=lambda: os.getenv("YOUTUBE_PRIVACY_STATUS", "public")
    )
    CONTAINS_SYNTHETIC_MEDIA: bool = Field(
        default_factory=lambda: os.getenv("CONTAINS_SYNTHETIC_MEDIA", "true").lower() in ("true", "1", "yes")
    )

    # External Binaries
    FFMPEG_PATH: str = Field(default_factory=_find_ffmpeg)
    FFPROBE_PATH: str = Field(default_factory=_find_ffprobe)

    def init_directories(self) -> None:
        """Ensure all required pipeline directories exist on disk."""
        for directory in [self.ASSETS_DIR, self.DATA_DIR, self.OUTPUT_DIR, self.TEMP_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        if not self.HISTORY_FILE.exists():
            self.HISTORY_FILE.write_text(json.dumps({"used_topics": []}, indent=2), encoding="utf-8")

        if not self.SPEND_TRACKER_FILE.exists():
            self.SPEND_TRACKER_FILE.write_text(
                json.dumps({"records": [], "total_spend_usd": 0.0}, indent=2), encoding="utf-8"
            )

    def validate_publishing_credentials(self) -> list[str]:
        """Check whether credentials necessary for uploading to YouTube are configured."""
        missing: list[str] = []
        if not self.YOUTUBE_CLIENT_ID:
            missing.append("YOUTUBE_CLIENT_ID")
        if not self.YOUTUBE_CLIENT_SECRET:
            missing.append("YOUTUBE_CLIENT_SECRET")
        if not self.YOUTUBE_REFRESH_TOKEN:
            missing.append("YOUTUBE_REFRESH_TOKEN")
        return missing

    def can_spend_fal(self) -> tuple[bool, str]:
        """
        Evaluate if Fal.ai video generation is permitted under configured hard budget caps.
        Returns (is_allowed, reason).
        """
        if not self.ENABLE_FAL_HOOK:
            return False, "ENABLE_FAL_HOOK is set to false."
        if not self.FAL_KEY:
            return False, "FAL_KEY is not configured."

        now = datetime.now(timezone.utc)
        current_date_str = now.strftime("%Y-%m-%d")
        current_month_str = now.strftime("%Y-%m")

        if not self.SPEND_TRACKER_FILE.exists():
            return True, "No prior spend recorded."

        try:
            with open(self.SPEND_TRACKER_FILE, "r", encoding="utf-8") as f:
                tracker = json.load(f)
        except Exception as e:
            logger.warning("Could not parse spend tracker (%s). Permitting with caution.", e)
            return True, "Tracker read failed; defaulting to allow."

        records: list[dict[str, Any]] = tracker.get("records", [])
        daily_spend = 0.0
        monthly_spend = 0.0

        for r in records:
            ts = r.get("timestamp", "")
            cost = float(r.get("cost_usd", 0.0))
            if ts.startswith(current_date_str):
                daily_spend += cost
            if ts.startswith(current_month_str):
                monthly_spend += cost

        cost_to_add = self.FAL_ESTIMATED_COST_PER_CALL
        if daily_spend + cost_to_add > self.FAL_DAILY_BUDGET_USD:
            return (
                False,
                f"Daily Fal budget reached (${daily_spend:.2f} + ${cost_to_add:.2f} > ${self.FAL_DAILY_BUDGET_USD:.2f}).",
            )
        if monthly_spend + cost_to_add > self.FAL_MONTHLY_BUDGET_USD:
            return (
                False,
                f"Monthly Fal budget reached (${monthly_spend:.2f} + ${cost_to_add:.2f} > ${self.FAL_MONTHLY_BUDGET_USD:.2f}).",
            )

        return True, "Within budget."

    def record_fal_spend(self, topic: str, cost: float | None = None) -> None:
        """Record an approved Fal.ai generation call in the spend tracker."""
        actual_cost = cost if cost is not None else self.FAL_ESTIMATED_COST_PER_CALL
        now_str = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": now_str,
            "service": "fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
            "cost_usd": round(actual_cost, 4),
            "topic": topic,
        }

        try:
            tracker: dict[str, Any] = {"records": [], "total_spend_usd": 0.0}
            if self.SPEND_TRACKER_FILE.exists():
                with open(self.SPEND_TRACKER_FILE, "r", encoding="utf-8") as f:
                    tracker = json.load(f)

            records = tracker.setdefault("records", [])
            records.append(record)
            tracker["total_spend_usd"] = round(sum(r.get("cost_usd", 0.0) for r in records), 4)

            with open(self.SPEND_TRACKER_FILE, "w", encoding="utf-8") as f:
                json.dump(tracker, f, indent=2)

            logger.info(
                "Logged Fal.ai spend: $%.2f (Total recorded: $%.2f)",
                actual_cost,
                tracker["total_spend_usd"],
            )
        except Exception as e:
            logger.error("Failed to write spend tracker: %s", e)


# Global singleton instance
config = PipelineConfig()
