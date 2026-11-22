#!/usr/bin/env python3
"""
One-Time YouTube OAuth 2.0 Refresh Token Generator.

This script runs a local desktop authorization flow using
google_auth_oauthlib.flow.InstalledAppFlow with the
'https://www.googleapis.com/auth/youtube.upload' scope.

It prints the generated YOUTUBE_REFRESH_TOKEN along with exact instructions
for saving it to your local .env file and GitHub Repository Secrets.

CRITICAL PRODUCTION REQUIREMENT:
--------------------------------
In Google Cloud Console (https://console.cloud.google.com/):
1. Navigate to 'APIs & Services' -> 'OAuth consent screen'.
2. If the Publishing Status is 'Testing', Google automatically expires
   OAuth refresh tokens after 7 DAYS!
3. Click 'PUBLISH APP' to switch the status to 'In Production'.
   In Production mode, your refresh token remains valid indefinitely,
   ensuring your automated daily GitHub Actions pipeline runs hands-free.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


def main() -> None:
    print("=" * 78)
    print(" YouTube Data API v3 - OAuth 2.0 Refresh Token Setup Helper")
    print("=" * 78)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("\n[ERROR] Required package 'google-auth-oauthlib' is not installed.")
        print("Please install dependencies first: pip install google-auth-oauthlib")
        sys.exit(1)

    # Load .env if present
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if len(v) >= 2 and (v[0] == v[-1] and v[0] in ('"', "'")):
                            v = v[1:-1]
                        if k not in os.environ and v and not v.startswith("your_"):
                            os.environ[k] = v
        except Exception:
            pass

    # Check for client secrets JSON or environment variables
    client_id = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    client_secrets_file = Path("client_secrets.json")
    if not client_secrets_file.exists():
        # Check parent directory or root
        alt_path = Path(__file__).resolve().parent.parent / "client_secrets.json"
        if alt_path.exists():
            client_secrets_file = alt_path

    flow: InstalledAppFlow | None = None

    if client_secrets_file.exists():
        print(f"\n[INFO] Found '{client_secrets_file}'. Loading OAuth credentials...")
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_file), scopes=SCOPES)
    elif client_id and client_secret:
        print("\n[INFO] Found YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in environment.")
        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost:8080/", "urn:ietf:wg:oauth:2.0:oob"],
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
    else:
        print("\n[SETUP REQUIRED] No credentials found.")
        print("You can provide credentials in one of two ways:")
        print("1. Download 'client_secrets.json' (Desktop App) from Google Cloud Console and place it here.")
        print("2. Or enter your OAuth Client ID and Secret interactively below.\n")

        try:
            input_client_id = input("Enter YOUTUBE_CLIENT_ID: ").strip()
            input_client_secret = input("Enter YOUTUBE_CLIENT_SECRET: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)

        if not input_client_id or not input_client_secret:
            print("[ERROR] Both Client ID and Client Secret are required.")
            sys.exit(1)

        client_config = {
            "installed": {
                "client_id": input_client_id,
                "client_secret": input_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost:8080/", "urn:ietf:wg:oauth:2.0:oob"],
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)

    print("\nStarting local authentication server...")
    print("Your default browser will open automatically to authenticate your YouTube channel account.")
    print("If prompted with 'Google hasn't verified this app', click 'Advanced' -> 'Go to <app name> (unsafe)'.\n")

    try:
        credentials = flow.run_local_server(
            port=8080,
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true",
        )
    except Exception as exc:
        print(f"\n[ERROR] Local server authentication failed: {exc}")
        print("Trying console flow as fallback...")
        credentials = flow.run_console(prompt="consent", access_type="offline")

    refresh_token = credentials.refresh_token
    if not refresh_token:
        print("\n[WARNING] No refresh token returned by Google.")
        print("This usually happens if offline access or prompt='consent' was omitted.")
        print("Please revoke access at https://myaccount.google.com/permissions and run this script again.")
        sys.exit(1)

    print("\n" + "=" * 78)
    print(" SUCCESS! YOUR OAUTH 2.0 REFRESH TOKEN IS READY:")
    print("=" * 78)
    print(f"\nYOUTUBE_REFRESH_TOKEN={refresh_token}\n")
    print("=" * 78)
    print(" NEXT STEPS:")
    print("=" * 78)
    print("1. Add this token to your local .env file:")
    print(f"   YOUTUBE_REFRESH_TOKEN={refresh_token}")
    print("\n2. Add it to your GitHub Repository Secrets (Settings -> Secrets and variables -> Actions):")
    print("   - Name:  YOUTUBE_REFRESH_TOKEN")
    print(f"   - Value: {refresh_token}")
    print("\n3. CRITICAL: In Google Cloud Console (APIs & Services -> OAuth consent screen):")
    print("   Ensure the 'Publishing Status' is changed from 'Testing' to 'In Production'.")
    print("   Otherwise, Google will invalidate this refresh token after exactly 7 days!")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
