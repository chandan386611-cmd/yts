"""
Official YouTube Data API v3 Resumable Uploader.

Refreshes OAuth 2.0 credentials via YOUTUBE_REFRESH_TOKEN, uploads final
vertical MP4 via resumable multipart chunks, enforces synthetic media tags,
and guarantees verification of uploaded video ID before reporting success.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

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

import json
import requests
from src.config import config
from src.script_generator import ShortScript

logger = logging.getLogger("youtube_uploader")


class YouTubeUploader:
    """Manages YouTube Data API v3 OAuth authentication and resumable publishing."""

    def __init__(self) -> None:
        pass

    def _get_authenticated_service(self) -> Any:
        """Construct authenticated YouTube API client using OAuth 2.0 refresh token."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError:
            return None

        missing = config.validate_publishing_credentials()
        if missing:
            raise ValueError(f"Missing required YouTube OAuth credentials: {', '.join(missing)}")

        logger.info("Refreshing YouTube OAuth 2.0 credentials via googleapiclient...")
        credentials = Credentials(
            token=None,
            refresh_token=config.YOUTUBE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.YOUTUBE_CLIENT_ID,
            client_secret=config.YOUTUBE_CLIENT_SECRET,
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
        )

        return build("youtube", "v3", credentials=credentials)

    def _refresh_access_token_direct(self) -> str:
        """Fetch fresh access token directly using OAuth refresh token."""
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": config.YOUTUBE_CLIENT_ID,
            "client_secret": config.YOUTUBE_CLIENT_SECRET,
            "refresh_token": config.YOUTUBE_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        }
        resp = requests.post(token_url, data=payload, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to refresh YouTube access token: HTTP {resp.status_code} - {resp.text}")
        data = resp.json()
        access_token = data.get("access_token")
        if not access_token:
            raise RuntimeError(f"No access_token returned by Google OAuth: {data}")
        return access_token

    def _upload_resumable_direct(self, video_path: Path, script: ShortScript) -> str:
        """Resumable upload using direct YouTube Data API v3 protocol over HTTP."""
        access_token = self._refresh_access_token_direct()
        file_size = video_path.stat().st_size

        metadata = {
            "snippet": {
                "title": script.title,
                "description": script.description,
                "tags": script.tags,
                "categoryId": "27",
                "defaultLanguage": "en",
                "defaultAudioLanguage": "en",
            },
            "status": {
                "privacyStatus": config.YOUTUBE_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": config.CONTAINS_SYNTHETIC_MEDIA,
                "embeddable": True,
                "license": "youtube",
            },
        }

        init_url = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(file_size),
        }

        logger.info("Initializing resumable upload session with YouTube API...")
        init_res = requests.post(init_url, headers=headers, json=metadata, timeout=30)
        if init_res.status_code != 200:
            raise RuntimeError(f"Failed to initiate resumable upload: HTTP {init_res.status_code} - {init_res.text}")

        upload_url = init_res.headers.get("Location")
        if not upload_url:
            raise RuntimeError("YouTube API did not return an upload session Location URL.")

        logger.info("Streaming video payload (%.2f MB)...", file_size / (1024 * 1024))
        with open(video_path, "rb") as f:
            upload_headers = {
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size),
            }
            upload_res = requests.put(upload_url, headers=upload_headers, data=f, timeout=180)

        if upload_res.status_code not in (200, 201):
            raise RuntimeError(f"YouTube upload failed: HTTP {upload_res.status_code} - {upload_res.text}")

        res_data = upload_res.json()
        video_id = res_data.get("id")
        if not video_id:
            raise RuntimeError(f"YouTube upload succeeded but no video ID returned: {res_data}")

        return str(video_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def upload_short(self, video_path: Path, script: ShortScript) -> str:
        """
        Executes resumable upload of final_short.mp4 to YouTube Shorts.

        Enforces:
          - privacyStatus (config.YOUTUBE_PRIVACY_STATUS)
          - selfDeclaredMadeForKids: False
          - containsSyntheticMedia: config.CONTAINS_SYNTHETIC_MEDIA
          - Explicit verification of returned video ID

        Returns:
            Uploaded YouTube Video ID (e.g. 'dQw4w9WgXcQ')
        """
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f"Video file {video_path} does not exist or is empty.")

        youtube = self._get_authenticated_service()

        if youtube is None:
            logger.info("google-api-python-client not found. Using direct HTTP resumable upload protocol.")
            video_id = self._upload_resumable_direct(video_path, script)
            watch_url = f"https://www.youtube.com/shorts/{video_id}"
            logger.info("=== YOUTUBE SHORT PUBLISHED SUCCESSFULLY ===")
            logger.info("Video ID: %s", video_id)
            logger.info("Short URL: %s", watch_url)
            return video_id

        from googleapiclient.http import MediaFileUpload

        body: dict[str, Any] = {
            "snippet": {
                "title": script.title,
                "description": script.description,
                "tags": script.tags,
                "categoryId": "27",  # Education / Psychological Science
                "defaultLanguage": "en",
                "defaultAudioLanguage": "en",
            },
            "status": {
                "privacyStatus": config.YOUTUBE_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": config.CONTAINS_SYNTHETIC_MEDIA,
                "embeddable": True,
                "license": "youtube",
            },
        }

        logger.info(
            "Initiating resumable upload for '%s' (Privacy: %s, Synthetic Media: %s, Size: %.2f MB)...",
            script.title,
            config.YOUTUBE_PRIVACY_STATUS,
            config.CONTAINS_SYNTHETIC_MEDIA,
            video_path.stat().st_size / (1024 * 1024),
        )

        # 5 MB resumable chunks
        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            chunksize=5 * 1024 * 1024,
            resumable=True,
        )

        insert_request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status:
                percent = int(status.progress() * 100)
                logger.info("Upload progress: %d%%", percent)

        video_id = response.get("id")
        if not video_id:
            raise RuntimeError(f"YouTube upload finished but returned no valid video ID: {response}")

        watch_url = f"https://www.youtube.com/shorts/{video_id}"
        logger.info("=== YOUTUBE SHORT PUBLISHED SUCCESSFULLY ===")
        logger.info("Video ID: %s", video_id)
        logger.info("Short URL: %s", watch_url)
        return str(video_id)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="Upload an existing video file directly to YouTube Shorts.")
    parser.add_argument("video_path", type=str, nargs="?", default="output/final_short.mp4", help="Path to MP4 video file (default: output/final_short.mp4)")
    parser.add_argument("--title", type=str, default="Unsolved Ancient History Mystery #Shorts #History", help="Video title")
    parser.add_argument("--description", type=str, default="Unsolved Ancient History Mysteries & Conspiracies. Watch till end! #Shorts #History #Mystery", help="Video description")
    parser.add_argument("--tags", nargs="*", default=["shorts", "history", "mystery", "conspiracy"], help="Video tags")

    args = parser.parse_args()
    target_path = Path(args.video_path)
    if not target_path.exists():
        print(f"Error: Video file '{target_path}' not found!")
        sys.exit(1)

    uploader = YouTubeUploader()
    script = ShortScript(
        topic=args.title,
        title=args.title,
        description=args.description,
        tags=args.tags,
        scenes=[],
    )
    print(f"Uploading {target_path} to YouTube Shorts...")
    vid_id = uploader.upload_short(target_path, script)
    print(f"\nSUCCESS! Video is live at: https://www.youtube.com/shorts/{vid_id}")
