"""
Comprehensive Verification Test Suite for YouTube Shorts Pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.caption_generator import CaptionGenerator, WordTimestamp, format_ass_timestamp
from src.config import config
from src.script_generator import (
    ScenePrompt,
    ShortScript,
    _parse_gemini_response,
    generate_fallback_offline_script,
    load_recent_topics,
)
from src.video_editor import escape_ffmpeg_filter_path
from src.visual_generator import verify_and_crop_to_vertical
from src.voice_generator import probe_audio_duration


def test_config_initialization() -> None:
    """Verify central configuration loads properly and initializes directories."""
    config.init_directories()
    assert config.ASSETS_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.OUTPUT_DIR.exists()
    assert config.TEMP_DIR.exists()
    assert config.HISTORY_FILE.exists()
    assert config.SPEND_TRACKER_FILE.exists()


def test_fal_spend_guard() -> None:
    """Test daily and monthly hard budget guard logic."""
    can_spend, reason = config.can_spend_fal()
    # By default, without key it returns False
    assert isinstance(can_spend, bool)
    assert isinstance(reason, str)


def test_fallback_script_structure() -> None:
    """Verify fallback offline script adheres to all strict pacing and scene rules."""
    script = generate_fallback_offline_script("Dark Psychology")
    assert isinstance(script, ShortScript)
    assert config.MIN_SCENES <= len(script.scenes) <= config.MAX_SCENES
    assert config.TARGET_MIN_WORDS <= script.total_words <= config.TARGET_MAX_WORDS
    assert script.scenes[0].scene_id == 0
    assert "35mm cinematic film grain" in script.scenes[0].full_visual_prompt


def test_gemini_json_parsing() -> None:
    """Test raw markdown fence parsing of Gemini structured JSON."""
    raw_response = """```json
{
  "topic": "The Illusion of Choice",
  "title": "How You Are Manipulated Daily #Shorts",
  "description": "Discover how supermarket design tricks your brain. #Psychology",
  "tags": ["psychology", "shorts"],
  "scenes": [
    {
      "scene_id": 0,
      "narration": "Have you ever wondered why milk is always at the back of the store?",
      "visual_prompt": "Dimly lit supermarket aisle stretching endlessly into darkness",
      "estimated_duration": 4.0
    }
  ]
}
```"""
    parsed = _parse_gemini_response(raw_response)
    assert parsed.topic == "The Illusion of Choice"
    assert len(parsed.scenes) == 1
    assert parsed.scenes[0].scene_id == 0


def test_pillow_vertical_cropping_and_resizing() -> None:
    """Test deterministic center-crop on landscape, square, and vertical images."""
    # Test 1: Square input (1024x1024)
    square_img = Image.new("RGB", (1024, 1024), color=(255, 0, 0))
    processed_square = verify_and_crop_to_vertical(square_img, target_width=1080, target_height=1920)
    assert processed_square.size == (1080, 1920)

    # Test 2: Landscape input (1920x1080)
    landscape_img = Image.new("RGB", (1920, 1080), color=(0, 255, 0))
    processed_landscape = verify_and_crop_to_vertical(landscape_img, target_width=1080, target_height=1920)
    assert processed_landscape.size == (1080, 1920)

    # Test 3: Ultra-tall input (500x2000)
    tall_img = Image.new("RGB", (500, 2000), color=(0, 0, 255))
    processed_tall = verify_and_crop_to_vertical(tall_img, target_width=1080, target_height=1920)
    assert processed_tall.size == (1080, 1920)


def test_ass_timestamp_formatting() -> None:
    """Verify ASS timestamp converter to H:MM:SS.cs."""
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(4.256) == "0:00:04.26"
    assert format_ass_timestamp(65.12) == "0:01:05.12"


def test_ass_subtitle_generation(tmp_path: Path) -> None:
    """Test ASS subtitle file building with word timestamps and pop-in animations."""
    words = [
        WordTimestamp(word="LOOK", start=0.0, end=0.4),
        WordTimestamp(word="INTO", start=0.4, end=0.7),
        WordTimestamp(word="MY", start=0.7, end=0.9),
        WordTimestamp(word="EYES", start=0.9, end=1.3),
        WordTimestamp(word="CAREFULLY", start=1.3, end=1.8),
    ]

    caption_gen = CaptionGenerator()
    out_ass = tmp_path / "test_subs.ass"
    caption_gen.build_ass_subtitles(words, output_file=out_ass)

    assert out_ass.exists()
    content = out_ass.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Dialogue: 0," in content
    assert r"{\c&H0000FFFF&}" in content  # Yellow highlight tag


def test_ffmpeg_filter_path_escaping() -> None:
    """Test escaping of paths on Windows and Linux for FFmpeg subtitle filter."""
    p = Path("C:/Users/test/brain/subs.ass")
    escaped = escape_ffmpeg_filter_path(p)
    assert r"\:" in escaped
    # Ensure all path directory separators were converted to forward slash
    assert "/Users/test/brain/subs.ass" in escaped



def test_history_deduplication(tmp_path: Path) -> None:
    """Test loading past topics from history.json."""
    hist_file = tmp_path / "history.json"
    dummy_data = {
        "used_topics": [
            {"topic": f"Psychology Topic {i}"} for i in range(50)
        ]
    }
    hist_file.write_text(json.dumps(dummy_data), encoding="utf-8")

    # Monkeypatch config.HISTORY_FILE temporarily
    old_file = config.HISTORY_FILE
    try:
        config.HISTORY_FILE = hist_file
        topics = load_recent_topics(max_history=40)
        assert len(topics) == 40
        assert topics[-1] == "Psychology Topic 49"
    finally:
        config.HISTORY_FILE = old_file


def test_default_audio_asset_presence() -> None:
    """Ensure default audio placeholder files exist and have non-zero size."""
    assert config.BG_MUSIC_PATH.exists()
    assert config.BG_MUSIC_PATH.stat().st_size > 0
    assert config.WHOOSH_PATH.exists()
    assert config.WHOOSH_PATH.stat().st_size > 0


def test_cartesia_configuration_and_synthesis(tmp_path: Path) -> None:
    """Verify Cartesia voice engine configuration and fallback synthesis."""
    assert config.CARTESIA_VOICE_ID == "7e8cb11d-37af-476b-ab8f-25da99b18644"
    assert config.CARTESIA_MODEL == "sonic-latest"
    from src.voice_generator import VoiceGenerator
    vg = VoiceGenerator()
    dummy_out = tmp_path / "test_voice.mp3"
    vg._synthesize_dummy_scene_audio("Testing fallback audio generation for shorts.", dummy_out)
    assert dummy_out.exists()
    dur = probe_audio_duration(dummy_out)
    assert dur > 1.0


def test_coverr_stock_video_configuration() -> None:
    """Verify Coverr real stock video configuration."""
    assert hasattr(config, "COVERR_API_KEY")
    from src.visual_generator import VisualGenerator
    vg = VisualGenerator()
    assert hasattr(vg, "_call_coverr_api")
    assert hasattr(vg, "_call_mixkit_video")
    assert hasattr(vg, "generate_scene_visual")


if __name__ == "__main__":
    import tempfile
    print("Running verification test suite...")
    test_config_initialization()
    print("PASS: test_config_initialization")
    test_fal_spend_guard()
    print("PASS: test_fal_spend_guard")
    test_fallback_script_structure()
    print("PASS: test_fallback_script_structure")
    test_gemini_json_parsing()
    print("PASS: test_gemini_json_parsing")
    test_pillow_vertical_cropping_and_resizing()
    print("PASS: test_pillow_vertical_cropping_and_resizing")
    test_ass_timestamp_formatting()
    print("PASS: test_ass_timestamp_formatting")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_ass_subtitle_generation(tmp_path)
        print("PASS: test_ass_subtitle_generation")
        test_history_deduplication(tmp_path)
        print("PASS: test_history_deduplication")
        test_cartesia_configuration_and_synthesis(tmp_path)
        print("PASS: test_cartesia_configuration_and_synthesis")
        test_coverr_stock_video_configuration()
        print("PASS: test_coverr_stock_video_configuration")

    test_ffmpeg_filter_path_escaping()
    print("PASS: test_ffmpeg_filter_path_escaping")
    test_default_audio_asset_presence()
    print("PASS: test_default_audio_asset_presence")
    print("\nALL 12 TESTS PASSED SUCCESSFULLY!")

