"""
Hybrid Visual Generator for YouTube Shorts.

Scene 0: Fal.ai Kling Video (5-second genuine AI video motion) with hard budget guards.
Scenes 1-N: Cloudflare Workers AI FLUX.1 Schnell still images deterministically
cropped and resized to 1080x1920 via Pillow.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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
from src.script_generator import ScenePrompt, ShortScript

logger = logging.getLogger("visual_generator")


def verify_and_crop_to_vertical(
    image_input: Image.Image | bytes | Path,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Image.Image:
    """
    Inspects image dimensions with Pillow, deterministically center-crops to 9:16 aspect ratio,
    and resizes using high-quality Lanczos resampling to exact target_width x target_height.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        img = image_input.convert("RGB")

    orig_w, orig_h = img.size
    target_ratio = target_width / target_height
    orig_ratio = orig_w / orig_h

    # Deterministic center-crop calculation
    if orig_ratio > target_ratio:
        # Image is too wide -> crop left and right
        crop_w = int(round(orig_h * target_ratio))
        offset_x = (orig_w - crop_w) // 2
        crop_box = (offset_x, 0, offset_x + crop_w, orig_h)
    elif orig_ratio < target_ratio:
        # Image is too tall -> crop top and bottom
        crop_h = int(round(orig_w / target_ratio))
        offset_y = (orig_h - crop_h) // 2
        crop_box = (0, offset_y, orig_w, offset_y + crop_h)
    else:
        crop_box = (0, 0, orig_w, orig_h)

    cropped = img.crop(crop_box)
    resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized


def generate_procedural_cinematic_fallback(
    scene_index: int,
    prompt: str,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Image.Image:
    """
    Generates a dark, dramatic, cinematic procedural placeholder image
    when AI image generation APIs are offline or unconfigured.
    """
    # Create base dark vignette gradient
    base = Image.new("RGB", (target_width, target_height), (12, 14, 20))
    draw = ImageDraw.Draw(base)

    # Varied dark hues for visual contrast across scenes
    hues = [
        ((15, 20, 35), (8, 10, 18)),     # Midnight blue
        ((35, 15, 20), (18, 8, 10)),     # Dark crimson
        ((25, 25, 20), (12, 12, 8)),     # Moody amber
        ((15, 30, 25), (8, 16, 12)),     # Forest emerald shadow
        ((30, 15, 35), (15, 8, 18)),     # Royal obsidian
    ]
    color_top, color_bottom = hues[scene_index % len(hues)]

    # Draw vertical linear gradient
    for y in range(target_height):
        factor = y / target_height
        r = int(color_top[0] * (1 - factor) + color_bottom[0] * factor)
        g = int(color_top[1] * (1 - factor) + color_bottom[1] * factor)
        b = int(color_top[2] * (1 - factor) + color_bottom[2] * factor)
        draw.line([(0, y), (target_width, y)], fill=(r, g, b))

    # Add subtle geometrical depth accents
    cx, cy = target_width // 2, target_height // 2
    offset = (scene_index * 60) % 300
    draw.ellipse(
        [cx - 350 - offset, cy - 350 + offset, cx + 350 + offset, cy + 350 - offset],
        outline=(255, 255, 255, 20),
        width=2,
    )
    draw.rectangle([100, 200, target_width - 100, target_height - 200], outline=(60, 70, 90), width=1)

    # Vignette shadow overlay
    vignette = Image.new("L", (target_width, target_height), 0)
    v_draw = ImageDraw.Draw(vignette)
    v_draw.ellipse([80, 80, target_width - 80, target_height - 80], fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(150))
    base.paste(Image.new("RGB", (target_width, target_height), (0, 0, 0)), mask=Image.eval(vignette, lambda a: 255 - a))

    return base


class VisualGenerator:
    """Manages Scene 0 video hook generation and Scenes 1-N FLUX image generation."""

    def __init__(self) -> None:
        self.visuals_dir = config.TEMP_DIR / "visuals"
        self.visuals_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=2, max=6),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _call_fal_kling_video(self, prompt: str, topic: str) -> Path | None:
        """
        Calls Fal.ai Kling 2.5 turbo API to generate 5-second 9:16 vertical video.
        Enforces spend budget checks before execution.
        """
        allowed, reason = config.can_spend_fal()
        if not allowed:
            logger.info("Fal.ai video generation skipped: %s. Using still image fallback.", reason)
            return None

        url = "https://queue.fal.run/fal-ai/kling-video/v2.5-turbo/pro/text-to-video"
        headers = {
            "Authorization": f"Key {config.FAL_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "aspect_ratio": "9:16",
            "duration": "5",
        }

        logger.info("Submitting Scene 0 prompt to Fal.ai Kling Video...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code not in (200, 201, 202):
            logger.warning("Fal.ai request submission failed (HTTP %d: %s)", response.status_code, response.text)
            return None

        res_data = response.json()
        video_url: str | None = None

        # Check if direct result or queue status
        if "video" in res_data and "url" in res_data["video"]:
            video_url = res_data["video"]["url"]
        elif "request_id" in res_data or "status_url" in res_data:
            # Poll queue
            status_url = res_data.get("status_url") or f"https://queue.fal.run/fal-ai/kling-video/v2.5-turbo/pro/requests/{res_data['request_id']}"
            max_wait_seconds = 180
            poll_interval = 5
            elapsed = 0

            logger.info("Polling Fal.ai video queue (timeout: %ds)...", max_wait_seconds)
            while elapsed < max_wait_seconds:
                time.sleep(poll_interval)
                elapsed += poll_interval
                status_res = requests.get(status_url, headers=headers, timeout=20)
                if status_res.status_code == 200:
                    status_json = status_res.json()
                    status_state = status_json.get("status")
                    if status_state == "COMPLETED":
                        video_url = status_json.get("response", {}).get("video", {}).get("url")
                        if not video_url and "video" in status_json:
                            video_url = status_json["video"].get("url")
                        break
                    elif status_state in ("FAILED", "CANCELLED"):
                        logger.error("Fal.ai video generation failed: %s", status_json.get("error"))
                        return None
                    else:
                        logger.debug("Fal.ai status: %s (%ds elapsed)", status_state, elapsed)
                else:
                    logger.warning("Fal status check HTTP %d", status_res.status_code)

        if not video_url:
            logger.warning("No video URL returned by Fal.ai.")
            return None

        # Download video to hook.mp4
        logger.info("Downloading Fal.ai video hook from %s...", video_url)
        hook_path = self.visuals_dir / "hook.mp4"
        vid_res = requests.get(video_url, timeout=60, stream=True)
        vid_res.raise_for_status()

        with open(hook_path, "wb") as f:
            for chunk in vid_res.iter_content(chunk_size=65536):
                f.write(chunk)

        # Record spend in tracker
        config.record_fal_spend(topic=topic)
        logger.info("Successfully downloaded Scene 0 Fal video hook to %s", hook_path)
        return hook_path

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=8),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _call_cloudflare_flux(self, prompt: str) -> bytes | None:
        """Call Cloudflare Workers AI FLUX.1 Schnell model for image generation."""
        if not config.CLOUDFLARE_ACCOUNT_ID or not config.CLOUDFLARE_API_TOKEN:
            return None

        url = f"https://api.cloudflare.com/client/v4/accounts/{config.CLOUDFLARE_ACCOUNT_ID}/ai/run/{config.CLOUDFLARE_FLUX_MODEL}"
        headers = {
            "Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=45)
        if response.status_code != 200:
            logger.warning("Cloudflare Workers AI HTTP %d: %s", response.status_code, response.text[:200])
            return None

        content_type = response.headers.get("content-type", "")
        # Raw binary image returned
        if "image" in content_type or "octet-stream" in content_type:
            return response.content

        # JSON response with base64 encoded image
        try:
            res_json = response.json()
            if "result" in res_json and "image" in res_json["result"]:
                b64_str = res_json["result"]["image"]
                return base64.b64decode(b64_str)
        except Exception as e:
            logger.warning("Error parsing Cloudflare JSON response: %s", e)

        return None

    def generate_scene_still_image(self, scene: ScenePrompt, index: int) -> Path:
        """
        Generate image for scene index via Cloudflare FLUX or procedural fallback,
        inspect pixel dimensions with Pillow, center-crop to 9:16, and resize to 1080x1920.
        """
        out_path = self.visuals_dir / f"scene_{index}.png"
        full_prompt = scene.full_visual_prompt

        logger.info("Generating visual for Scene %d: '%s'...", index, scene.visual_prompt[:60])
        raw_bytes: bytes | None = None

        if config.CLOUDFLARE_ACCOUNT_ID and config.CLOUDFLARE_API_TOKEN:
            try:
                raw_bytes = self._call_cloudflare_flux(full_prompt)
            except Exception as e:
                logger.warning("Cloudflare FLUX call failed for scene %d: %s", index, e)

        if raw_bytes:
            processed_img = verify_and_crop_to_vertical(
                raw_bytes,
                target_width=config.VIDEO_WIDTH,
                target_height=config.VIDEO_HEIGHT,
            )
        else:
            logger.info("Using procedural cinematic visual fallback for Scene %d", index)
            processed_img = generate_procedural_cinematic_fallback(
                scene_index=index,
                prompt=scene.visual_prompt,
                target_width=config.VIDEO_WIDTH,
                target_height=config.VIDEO_HEIGHT,
            )

        processed_img.save(out_path, format="PNG", quality=95)
        logger.info("Saved verified 1080x1920 image to %s", out_path)
        return out_path

    def generate_all_visuals(self, script: ShortScript) -> tuple[Path | None, list[Path]]:
        """
        Generates all visual assets for the Short.

        Returns:
            (hook_video_path_or_none, list_of_scene_still_images)
            If Scene 0 AI video generation succeeds, hook_video_path is set to hook.mp4.
            If Scene 0 AI video fails or is disabled, hook_video_path is None, and
            scene_0.png is included in the still images list.
        """
        hook_video_path: Path | None = None
        scene_stills: list[Path] = []

        # Scene 0: Attempt genuine AI video motion via Fal.ai Kling Video
        scene_0 = script.scenes[0]
        if config.ENABLE_FAL_HOOK and config.FAL_KEY:
            try:
                hook_video_path = self._call_fal_kling_video(
                    prompt=scene_0.full_visual_prompt,
                    topic=script.topic,
                )
            except Exception as e:
                logger.warning("Fal.ai video hook generation failed: %s. Falling back to animated still.", e)
                hook_video_path = None

        # If Scene 0 video hook was not created, generate a still image for Scene 0
        start_idx = 1
        if hook_video_path is None:
            logger.info("Scene 0 will use camera-animated still image.")
            img_0 = self.generate_scene_still_image(scene_0, 0)
            scene_stills.append(img_0)
        else:
            # Placeholder still in case of video inspection fallback
            self.generate_scene_still_image(scene_0, 0)

        # Scenes 1 to N: Generate still images
        for idx in range(start_idx, len(script.scenes)):
            scene = script.scenes[idx]
            img_path = self.generate_scene_still_image(scene, idx)
            scene_stills.append(img_path)

        return hook_video_path, scene_stills
