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
import re
import time
import urllib.parse
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


ABSTRACT_SEARCH_STOPWORDS: set[str] = {
    # Moods & Emotions (cause random unrelated stock clips)
    "sad", "happy", "crying", "angry", "loyal", "loyalty", "emotional", "heartbreaking",
    "heartbroken", "intense", "dramatic", "crazy", "shocking", "creepy", "scary",
    "terrifying", "mindblowing", "beautiful", "amazing", "incredible", "satisfying",
    "mysterious", "secret", "secrets", "mystery", "curious", "strange", "impossible",
    # Abstract concepts & Non-physical traits
    "poor", "rich", "real", "fake", "true", "truth", "good", "bad", "deep",
    "best", "worst", "dark", "light", "pure", "weird", "unbelievable",
    # Time, Question & Generic filler words
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "today", "yesterday", "tomorrow", "forever", "never", "always",
    "first", "last", "every", "daily", "year", "years", "night", "days",
    "story", "stories", "facts", "fact", "life", "reason", "time", "human",
    "cinematic", "portrait", "shot", "macro", "view", "scene", "footage", "video",
}


class VisualGenerator:
    """Manages Scene 0 video hook generation and Scenes 1-N FLUX image generation."""

    def __init__(self) -> None:
        self.visuals_dir = config.TEMP_DIR / "visuals"
        self.visuals_dir.mkdir(parents=True, exist_ok=True)

    def _call_fal_flux(self, prompt: str) -> bytes | None:
        """Call Fal.ai FLUX.1 Schnell for instantaneous 1080x1920 vertical cinematic images."""
        if not config.FAL_KEY:
            return None
        url = "https://fal.run/fal-ai/flux/schnell"
        headers = {
            "Authorization": f"Key {config.FAL_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "image_size": {"width": 1080, "height": 1920},
            "num_inference_steps": 4,
            "enable_safety_checker": False,
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                images = data.get("images", [])
                if images and "url" in images[0]:
                    img_dl = requests.get(images[0]["url"], timeout=20)
                    if img_dl.status_code == 200 and len(img_dl.content) > 10000:
                        logger.info("Retrieved authentic 1080x1920 Fal FLUX visual")
                        return img_dl.content
        except Exception as e:
            logger.debug("Fal FLUX call failed: %s", e)
        return None

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

        url = "https://queue.fal.run/fal-ai/kling-video/v1/turbo/text-to-video"
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
        elif "request_id" in res_data or "status_url" in res_data or "response_url" in res_data:
            # Poll queue
            status_url = res_data.get("status_url") or f"https://queue.fal.run/fal-ai/kling-video/v1/turbo/requests/{res_data['request_id']}/status"
            response_url = res_data.get("response_url") or f"https://queue.fal.run/fal-ai/kling-video/v1/turbo/requests/{res_data['request_id']}"
            max_wait_seconds = 180
            poll_interval = 5
            elapsed = 0

            logger.info("Polling Fal.ai video queue (timeout: %ds)...", max_wait_seconds)
            while elapsed < max_wait_seconds:
                time.sleep(poll_interval)
                elapsed += poll_interval
                status_res = requests.get(status_url, headers=headers, timeout=20)
                if status_res.status_code in (200, 202):
                    status_json = status_res.json()
                    status_state = status_json.get("status")
                    if status_state == "COMPLETED":
                        resp_fetch = requests.get(response_url, headers=headers, timeout=20)
                        if resp_fetch.status_code == 200:
                            video_url = resp_fetch.json().get("video", {}).get("url")
                        if not video_url:
                            video_url = status_json.get("response", {}).get("video", {}).get("url")
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

    def _call_coverr_api(self, query: str, output_path: Path) -> Path | None:
        """Fetch 1080p real stock video clip matching query from Coverr API."""
        coverr_key = getattr(config, "COVERR_API_KEY", "")
        if not coverr_key:
            return None

        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_q:
            return None

        # Build progressive fallback search terms prioritizing concrete physical nouns
        raw_tokens = [w.lower() for w in clean_q.split() if len(w) >= 3]
        concrete_tokens = [w for w in raw_tokens if w not in ABSTRACT_SEARCH_STOPWORDS]

        candidate_queries = [clean_q]
        # Remove abstract adjectives so Coverr gets concrete multi-word queries (e.g. 'dog hospital' instead of 'sad dog hospital')
        if concrete_tokens and " ".join(concrete_tokens) != clean_q.lower():
            candidate_queries.append(" ".join(concrete_tokens))

        if len(concrete_tokens) >= 2:
            candidate_queries.append(" ".join(concrete_tokens[:2]))

        # Fallback to single tokens ONLY if they are concrete physical nouns (NEVER search abstract words like 'sad' or 'loyal'!)
        for t in concrete_tokens:
            if t not in candidate_queries:
                candidate_queries.append(t)

        url = "https://api.coverr.co/videos"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        for q in candidate_queries:
            params = {
                "query": q,
                "page_size": 4,
                "urls": "true",
                "api_key": coverr_key,
            }
            try:
                res = requests.get(url, headers=headers, params=params, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    hits = data.get("hits", [])
                    # Rank hits: prioritize native vertical (9:16) and full 1080p height
                    def _rank_coverr_hit(h: dict[str, Any]) -> int:
                        score = 0
                        if h.get("is_vertical") is True or h.get("aspect_ratio") in ("9:16", "vertical"):
                            score += 100
                        if h.get("max_height") and h.get("max_height") >= 1080:
                            score += 50
                        return score

                    sorted_hits = sorted(hits, key=_rank_coverr_hit, reverse=True)
                    for hit in sorted_hits:
                        urls = hit.get("urls", {})
                        video_url = urls.get("mp4") or urls.get("mp4_download")
                        if video_url:
                            dl = requests.get(video_url, stream=True, timeout=25)
                            if dl.status_code == 200:
                                with open(output_path, "wb") as f:
                                    for chunk in dl.iter_content(chunk_size=16384):
                                        f.write(chunk)
                                if output_path.exists() and output_path.stat().st_size > 50000:
                                    is_vert = hit.get("is_vertical") is True or hit.get("aspect_ratio") in ("9:16", "vertical")
                                    logger.info(
                                        "Retrieved real 1080p Coverr video for '%s' ('%s', %s, %.2f MB)",
                                        q,
                                        hit.get("title", "clip")[:35],
                                        "NATIVE VERTICAL" if is_vert else "HORIZONTAL HD",
                                        output_path.stat().st_size / (1024 * 1024),
                                    )
                                    return output_path
                elif res.status_code in (401, 403, 429):
                    logger.warning("Coverr API HTTP %d: %s", res.status_code, res.text[:100])
                    break
            except Exception as e:
                logger.debug("Coverr fetch error for '%s': %s", q, e)

        return None

    def _call_mixkit_video(self, query: str, output_path: Path) -> Path | None:
        """Fetch HD stock video clip from Mixkit (Envato) public CDN."""
        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_q:
            return None

        # Build candidate tags prioritizing concrete physical subjects
        raw_tokens = [w.lower() for w in clean_q.split() if len(w) >= 3]
        concrete_tokens = [w for w in raw_tokens if w not in ABSTRACT_SEARCH_STOPWORDS]

        candidate_tags = []
        if len(concrete_tokens) >= 2:
            candidate_tags.append("-".join(concrete_tokens[:2]))
        for t in concrete_tokens:
            if t not in candidate_tags:
                candidate_tags.append(t)
        # If no concrete token found, fall back to non-stopword tokens
        if not candidate_tags and raw_tokens:
            candidate_tags.append(raw_tokens[0])

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        for tag in candidate_tags:
            search_url = f"https://mixkit.co/free-stock-video/{tag}/"
            try:
                res = requests.get(search_url, headers=headers, timeout=10)
                if res.status_code != 200:
                    continue

                links = re.findall(r"https://assets\.mixkit\.co/videos/(\d+)/\1-[a-zA-Z0-9_\-\.]+\.mp4", res.text)
                if not links:
                    raw_links = re.findall(r"https://assets\.mixkit\.co/videos/[a-zA-Z0-9_\-\./]+?\.mp4", res.text)
                    if raw_links:
                        video_url = raw_links[0]
                    else:
                        continue
                else:
                    video_id = links[0]
                    video_url = f"https://assets.mixkit.co/videos/{video_id}/{video_id}-720.mp4"

                dl = requests.get(video_url, stream=True, headers=headers, timeout=25)
                if dl.status_code != 200 and "video_id" in locals():
                    video_url = f"https://assets.mixkit.co/videos/{video_id}/{video_id}-360.mp4"
                    dl = requests.get(video_url, stream=True, headers=headers, timeout=25)

                if dl.status_code == 200:
                    with open(output_path, "wb") as f:
                        for chunk in dl.iter_content(chunk_size=16384):
                            f.write(chunk)
                    if output_path.exists() and output_path.stat().st_size > 50000:
                        logger.info(
                            "Retrieved real HD Mixkit video for '%s' (%.2f MB)",
                            tag,
                            output_path.stat().st_size / (1024 * 1024),
                        )
                        return output_path
            except Exception as e:
                logger.debug("Mixkit fetch error for '%s': %s", tag, e)

        return None

    def _call_pexels_api(self, query: str) -> bytes | None:
        """Fetch vertical high-res photography matching query from Pexels API."""
        pexels_key = getattr(config, "PEXELS_API_KEY", "")
        if not pexels_key:
            return None

        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_q:
            return None

        raw_tokens = [w.lower() for w in clean_q.split() if len(w) >= 3]
        concrete_tokens = [w for w in raw_tokens if w not in ABSTRACT_SEARCH_STOPWORDS]

        candidate_queries = [clean_q]
        if concrete_tokens and " ".join(concrete_tokens) != clean_q.lower():
            candidate_queries.append(" ".join(concrete_tokens))
        if len(concrete_tokens) >= 2:
            candidate_queries.append(" ".join(concrete_tokens[:2]))

        url = "https://api.pexels.com/v1/search"
        headers = {
            "Authorization": pexels_key,
            "User-Agent": "YouTubeShortsPipeline/1.0",
        }

        for q in candidate_queries:
            params = {
                "query": q,
                "orientation": "portrait",
                "per_page": 5,
            }
            try:
                res = requests.get(url, headers=headers, params=params, timeout=12)
                if res.status_code == 200:
                    data = res.json()
                    photos = data.get("photos", [])
                    if photos:
                        img_url = (
                            photos[0].get("src", {}).get("portrait")
                            or photos[0].get("src", {}).get("large2x")
                            or photos[0].get("src", {}).get("original")
                        )
                        if img_url:
                            dl = requests.get(img_url, timeout=20)
                            if dl.status_code == 200 and len(dl.content) > 10000:
                                logger.info("Found Pexels portrait photo for '%s'", q)
                                return dl.content
                elif res.status_code in (401, 403, 429):
                    logger.warning("Pexels API HTTP %d: %s", res.status_code, res.text[:150])
                    break
            except Exception as e:
                logger.debug("Pexels fetch error for '%s': %s", q, e)
        return None

    def _call_wikimedia_commons(self, query: str, topic: str = "") -> bytes | None:
        """
        Fetch high-resolution real photograph from Wikimedia Commons API.
        100% free, unlimited, public domain / Creative Commons archive.
        Cascades through candidate query simplifications to ensure 100% match rate.
        """
        if not query or not query.strip():
            return None

        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        words = clean_q.split()

        candidates: list[str] = []
        if len(words) >= 3:
            candidates.append(" ".join(words[:3]))
        if len(words) >= 2:
            candidates.append(" ".join(words[:2]))
        if words and len(words[0]) >= 4:
            candidates.append(words[0])
        if topic:
            clean_topic = re.sub(r"[^\w\s]", " ", topic).strip()
            topic_words = clean_topic.split()
            if len(topic_words) >= 2:
                candidates.append(" ".join(topic_words[:2]))

        headers = {
            "User-Agent": "YouTubeShortsPipeline/1.0 (contact@zorvian.media)",
        }

        for search_term in candidates:
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": search_term,
                "gsrnamespace": 6,
                "gsrlimit": 5,
                "prop": "imageinfo",
                "iiprop": "url|mime|size|width|height",
                "format": "json",
            }
            try:
                res = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=headers, timeout=8)
                if res.status_code != 200:
                    continue

                pages = res.json().get("query", {}).get("pages", {})
                for _, page in pages.items():
                    info_list = page.get("imageinfo", [])
                    if not info_list:
                        continue
                    info = info_list[0]
                    url = info.get("url")
                    mime = info.get("mime", "")
                    w = info.get("width", 0)
                    h = info.get("height", 0)

                    # Filter for quality JPEG/PNG images >= 600px
                    if url and ("image/jpeg" in mime or "image/png" in mime) and (w >= 600 or h >= 600):
                        dl = requests.get(url, headers=headers, timeout=15)
                        if dl.status_code == 200 and len(dl.content) > 15000:
                            logger.info("Found high-res Wikimedia photograph for '%s' (%dx%d)", search_term, w, h)
                            return dl.content
            except Exception as e:
                logger.debug("Wikimedia search failed for '%s': %s", search_term, e)
        return None

    def _call_pollinations_flux(self, prompt: str) -> bytes | None:
        """Call Pollinations.ai free FLUX image generation as fallback."""
        try:
            clean_prompt = prompt[:200]
            encoded = urllib.parse.quote(clean_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux"
            res = requests.get(url, timeout=35)
            if res.status_code == 200 and len(res.content) > 10000:
                logger.info("Generated image via Pollinations.ai FLUX")
                return res.content
            elif res.status_code == 402:
                logger.debug("Pollinations rate limit (45s) active.")
        except Exception as e:
            logger.debug("Pollinations call failed: %s", e)
        return None

    def generate_scene_still_image(self, scene: ScenePrompt, index: int, topic: str = "") -> Path:
        """
        Multi-tier image generator:
        1. Cloudflare Workers AI FLUX (if available and within daily neuron quota)
        2. Pexels Stock API (if PEXELS_API_KEY is configured)
        3. Wikimedia Commons high-resolution photography archive (100% free, unlimited)
        4. Pollinations.ai FLUX fallback
        5. Procedural cinematic fallback (only if offline/all APIs fail)
        """
        out_path = self.visuals_dir / f"scene_{index}.png"
        full_prompt = scene.full_visual_prompt
        search_query = getattr(scene, "image_search_query", "") or scene.visual_prompt[:40]

        logger.info("Generating visual for Scene %d (Query: '%s')...", index, search_query)
        raw_bytes: bytes | None = None

        # Tier 1: Fal.ai FLUX (High-speed, photorealistic 1080x1920 vertical composition)
        if config.FAL_KEY:
            try:
                raw_bytes = self._call_fal_flux(full_prompt)
            except Exception as e:
                logger.debug("Fal FLUX call failed for scene %d: %s", index, e)

        # Tier 2: Cloudflare Workers AI FLUX
        if not raw_bytes and config.CLOUDFLARE_ACCOUNT_ID and config.CLOUDFLARE_API_TOKEN:
            try:
                raw_bytes = self._call_cloudflare_flux(full_prompt)
            except Exception as e:
                logger.warning("Cloudflare FLUX call failed for scene %d: %s", index, e)

        # Tier 3: Pexels API
        if not raw_bytes and getattr(config, "PEXELS_API_KEY", ""):
            try:
                raw_bytes = self._call_pexels_api(search_query)
            except Exception as e:
                logger.debug("Pexels fallback failed for scene %d: %s", index, e)

        # Tier 4: Wikimedia Commons High-Res Archive (Free, authentic photography)
        if not raw_bytes:
            try:
                raw_bytes = self._call_wikimedia_commons(search_query, topic=topic)
            except Exception as e:
                logger.debug("Wikimedia fallback failed for scene %d: %s", index, e)

        # Tier 5: Pollinations.ai FLUX
        if not raw_bytes:
            try:
                raw_bytes = self._call_pollinations_flux(scene.visual_prompt)
            except Exception as e:
                logger.debug("Pollinations fallback failed for scene %d: %s", index, e)

        if raw_bytes:
            processed_img = verify_and_crop_to_vertical(
                raw_bytes,
                target_width=config.VIDEO_WIDTH,
                target_height=config.VIDEO_HEIGHT,
            )
            logger.info("Retrieved authentic photo for Scene %d", index)
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

    def generate_scene_visual(self, scene: ScenePrompt, index: int, topic: str = "") -> Path:
        """
        Multi-tier visual generator prioritizing real moving stock video clips:
        Tier 1: Coverr Stock Video API (1080p MP4 real footage)
        Tier 2: Mixkit Stock Video CDN (HD MP4 real footage)
        Tier 3: Cloudflare Workers AI FLUX (still image)
        Tier 4: Pexels API (still image)
        Tier 5: Wikimedia Commons High-Res Archive (still image)
        Tier 6: Pollinations.ai FLUX (still image)
        Tier 7: Procedural cinematic fallback (still image)
        """
        video_out = self.visuals_dir / f"scene_{index}.mp4"
        search_query = getattr(scene, "image_search_query", "") or scene.visual_prompt[:40]

        logger.info("Generating visual for Scene %d (Query: '%s')...", index, search_query)

        # For gaming, space, sci-fi, CGI, and hyper-specific creative topics, stock footage libraries (Coverr/Mixkit)
        # return irrelevant generic clips (e.g. random sunbathers on a beach or office workers).
        # Bypass stock footage and use Fal.ai FLUX 1080x1920 vertical photorealistic generation.
        is_gaming_or_fiction = any(k in (topic + " " + search_query + " " + scene.visual_prompt).lower() for k in [
            "gta", "game", "gaming", "rockstar", "vice", "leonida", "budget",
            "character", "animation", "3d", "cgi", "developer", "indian", "studio", "bangalore",
            "sun", "space", "earth", "planet", "galaxy", "orbit", "solar", "astronomy",
            "universe", "disappear", "frozen", "darkness", "cosmos", "moon", "star", "what if"
        ])

        if not is_gaming_or_fiction:
            # Tier 1: Coverr Real Video
            if config.COVERR_API_KEY:
                try:
                    v_path = self._call_coverr_api(search_query, video_out)
                    if v_path and v_path.exists() and v_path.stat().st_size > 50000:
                        return v_path
                except Exception as e:
                    logger.debug("Coverr video fetch error for scene %d: %s", index, e)

            # Tier 2: Mixkit Real Video
            try:
                v_path = self._call_mixkit_video(search_query, video_out)
                if v_path and v_path.exists() and v_path.stat().st_size > 50000:
                    return v_path
            except Exception as e:
                logger.debug("Mixkit video fetch error for scene %d: %s", index, e)

        # Tier 3-7: Still Image Fallback with Ken Burns (uses Fal.ai FLUX as Tier 1)
        img_path = self.generate_scene_still_image(scene, index, topic=topic)
        return img_path

    def generate_all_visuals(self, script: ShortScript) -> list[Path]:
        """
        Generates all visual assets (real moving video clips or still images) for the Short.

        Returns:
            list[Path]: List of video (.mp4) or image (.png) paths matching each scene in script.
        """
        scene_visuals: list[Path] = []

        # Scene 0: Attempt genuine AI video motion via Fal.ai Kling Video if configured
        scene_0 = script.scenes[0]
        hook_video_path: Path | None = None
        if config.ENABLE_FAL_HOOK and config.FAL_KEY:
            try:
                hook_video_path = self._call_fal_kling_video(
                    prompt=scene_0.full_visual_prompt,
                    topic=script.topic,
                )
            except Exception as e:
                logger.warning("Fal.ai video hook generation failed: %s. Using stock video.", e)
                hook_video_path = None

        if hook_video_path and hook_video_path.exists():
            scene_visuals.append(hook_video_path)
            start_idx = 1
        else:
            start_idx = 0

        # Generate real moving video clips (or still fallbacks) for all remaining scenes
        for idx in range(start_idx, len(script.scenes)):
            scene = script.scenes[idx]
            visual_path = self.generate_scene_visual(scene, idx, topic=script.topic)
            scene_visuals.append(visual_path)

        return scene_visuals
