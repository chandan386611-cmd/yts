# Autonomous YouTube Shorts Generation & Publishing Pipeline

A production-grade, cost-capped, fully autonomous vertical (1080x1920, 30fps) YouTube Shorts creation and publishing engine in Python 3.11+. The system runs hands-free on a daily schedule via GitHub Actions, producing high-retention 40–45 second shorts using a hybrid AI architecture.

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │   GitHub Actions Cron    │ (Daily 12:00 UTC / 5:30 PM IST)
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │     src/config.py        │
                          │ Hard Budget & Art Locks  │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  src/script_generator.py │ ◄── Deduplicates against data/history.json
                          │ Google Gemini (105-120w) │
                          └─────────────┬────────────┘
                                        │
                                        ▼
            ┌────────────────────────────────────────────────────────┐
            │                 src/voice_generator.py                 │
            │   Tier 1: Cartesia Sonic (sonic-latest, ultra-human)   │
            │   Tier 2: ElevenLabs (eleven_flash_v2_5)               │
            │   Tier 3: Microsoft Edge-TTS (+14% speed boost)        │
            └───────────────────────────┬────────────────────────────┘
                                        │
                             [ffprobe Duration Gate]
                               (40.0s <= D <= 45.0s)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            [Outside Window]                         [Within Window]
         Gemini Feedback Re-prompt                          │
           (Expand / Compress)                              │
           (Up to 3 Attempts)                               │
                                                            ▼
                                          ┌───────────────────────────────────┐
                                          │      src/visual_generator.py      │
                                          ├───────────────────────────────────┤
                                          │ Scene 0 (5s Hook Video Motion):   │
                                          │   Fal.ai Kling Video Turbo        │
                                          │   * Hard Daily/Monthly Spend Cap  │
                                          │   * Clean fallback to FLUX still  │
                                          ├───────────────────────────────────┤
                                          │ Scenes 1-N (35-40s Stills):       │
                                          │   Cloudflare FLUX.1 Schnell       │
                                          │   * Verified & cropped via Pillow │
                                          │   * Exact 1080x1920 center-crop   │
                                          └─────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                          ┌───────────────────────────────────┐
                                          │     src/caption_generator.py      │
                                          │ faster-whisper (int8 CPU) -> .ass │
                                          │ 2-3 word chunks, yellow pop-in    │
                                          └─────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                          ┌───────────────────────────────────┐
                                          │        src/video_editor.py        │
                                          │ FFmpeg Ken Burns (zoompan)        │
                                          │ Audio ducking (-22dB) + whoosh SFX│
                                          │ Subtitle burn -> final_short.mp4  │
                                          └─────────────────┬─────────────────┘
                                                            │
                                      ┌─────────────────────┴─────────────────────┐
                                      │                                           │
                             [--dry-run Flag]                               [Production]
                                      │                                           │
                               Render local video                        src/youtube_uploader.py
                            Do NOT touch YouTube API                     Official YouTube API v3
                            Do NOT touch history.json                    Resumable chunked upload
                                                                                  │
                                                                           [Verify Video ID]
                                                                                  │
                                                                      Update data/history.json
                                                                      Commit & push via GitHub Actions
```

---

## Directory Structure

```
├── .github/
│   └── workflows/
│       └── daily_short.yml          # GitHub Actions scheduled workflow
├── assets/
│   ├── bg_music.mp3                 # Ambient royalty-free background track
│   └── whoosh.mp3                   # Scene transition SFX
├── data/
│   ├── history.json                 # Used topics and published video records
│   └── spend_tracker.json           # Daily & monthly Fal.ai API spending log
├── output/
│   └── final_short.mp4              # Rendered vertical Short (1080x1920, 30fps)
├── scripts/
│   ├── generate_default_assets.py   # Synthesizes placeholder audio if assets are missing
│   └── get_youtube_refresh_token.py # One-time desktop OAuth 2.0 setup tool
├── src/
│   ├── __init__.py
│   ├── config.py                    # Central settings, constants, and budget guards
│   ├── script_generator.py          # Gemini AI script generator + duration re-prompter
│   ├── voice_generator.py           # Cartesia Sonic / ElevenLabs / Edge-TTS audio & duration gate
│   ├── visual_generator.py          # Fal.ai Kling hook + Cloudflare FLUX stills
│   ├── caption_generator.py         # faster-whisper transcription + dynamic ASS subtitles
│   ├── video_editor.py              # FFmpeg zoompan, audio ducking, and subtitle burn-in
│   ├── youtube_uploader.py          # Official YouTube Data API v3 resumable uploader
│   └── main.py                      # CLI entrypoint and pipeline orchestrator
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py             # Test suite
├── .env.example                     # Environment configuration template
├── .gitignore
├── requirements.txt                 # Project dependencies
└── README.md
```

---

## Key Features & Constraints

### 1. Hybrid AI Cost Optimization
- **Scene 0 (5s Opening Hook)**: Calls `fal-ai/kling-video/v2.5-turbo/pro/text-to-video` (9:16 aspect ratio) for real AI camera motion.
- **Budget Protection**: Configurable hard daily (`FAL_DAILY_BUDGET_USD=1.00`) and monthly (`FAL_MONTHLY_BUDGET_USD=25.00`) caps. If reached, Scene 0 seamlessly falls back to a FLUX still with Ken Burns animation.
- **Scenes 1 to N (Remaining 35–40s)**: Generated via Cloudflare Workers AI (`@cf/black-forest-labs/flux-1-schnell`), center-cropped and resized with Pillow to exact `1080x1920`.

### 2. Strict Audio Duration Gate (40.0s – 45.0s)
- Total spoken narration word count targets **105 to 120 words** across **9 to 11 scenes**.
- After voiceover generation, `ffprobe` validates that total audio duration is strictly between `40.0` and `45.0` seconds.
- If duration falls outside this window, the pipeline automatically re-prompts Gemini with explicit expansion/compression feedback and regenerates audio (up to 3 attempts) **before spending any Fal.ai video credits**.

### 3. Voiceover & Edge-TTS Fallback
- Primary voiceover uses ElevenLabs `eleven_flash_v2_5` (cost-effective at 0.5 credits/char).
- If ElevenLabs quota is exhausted, authentication fails, or `USE_EDGE_TTS_FALLBACK=true`, the system automatically switches to Microsoft Edge TTS (`en-US-ChristopherNeural`) without crashing.

### 4. Dynamic Styled Captions
- Audio is transcribed with `faster-whisper` (`base.en` on CPU, `int8` quantization).
- Captions are formatted into punchy 2–3 word chunks and written to an `.ass` subtitle file.
- Centers screen above YouTube UI overlays, styling key words in **bold yellow** (`#FFFF00`) and secondary words in **white** (`#FFFFFF`) with a dynamic bounce scale pop-in animation (`\fscx115\fscy115 -> \fscx100\fscy100`).

### 5. Ken Burns Animation & Frame-Accurate Video Assembly
- Each still image is rendered into a video clip with FFmpeg `zoompan` alternating between 6 varied camera motions (zoom-in, zoom-out, pan up, pan down, pan left, pan right).
- Clip duration matches the exact `ffprobe` duration of that scene's audio segment, preventing any audio-video desync.
- Ambient background music is ducked to `-22dB` under the narration, and subtle whoosh SFX (`assets/whoosh.mp3`) are inserted at scene cuts.

---

## Local Setup & Quickstart

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create and activate Python virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate royalty-free default placeholder audio assets
python scripts/generate_default_assets.py
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (from Google AI Studio) |
| `ELEVENLABS_API_KEY` | ElevenLabs API key (optional if using Edge-TTS) |
| `USE_EDGE_TTS_FALLBACK` | Set `true` to enable automatic Edge-TTS fallback |
| `FAL_KEY` | Fal.ai API key for Scene 0 Kling video motion |
| `FAL_DAILY_BUDGET_USD` | Daily hard limit in USD for Fal.ai (default: `1.00`) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare Workers AI Account ID |
| `CLOUDFLARE_API_TOKEN` | Cloudflare Workers AI API Token |
| `YOUTUBE_CLIENT_ID` | Google OAuth 2.0 Client ID |
| `YOUTUBE_CLIENT_SECRET` | Google OAuth 2.0 Client Secret |
| `YOUTUBE_REFRESH_TOKEN` | Generated via `scripts/get_youtube_refresh_token.py` |
| `NICHE_TOPIC` | Channel niche (default: `Dark Psychology & Human Behavior Facts`) |

---

## YouTube OAuth 2.0 Setup (One-Time)

To authorize the pipeline to publish directly to your YouTube channel:

1. **Google Cloud Console Setup**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/).
   - Create a project and enable the **YouTube Data API v3**.
   - Navigate to **APIs & Services > Credentials**.
   - Create an **OAuth 2.0 Client ID** (Application type: **Desktop App**).
   - Copy the Client ID and Client Secret.

2. > [!CRITICAL]
   > ### **Production OAuth Consent Screen Setting**
   > In Google Cloud Console under **APIs & Services > OAuth consent screen**:
   > If the Publishing Status is set to **Testing**, Google will automatically expire your refresh token after **7 days**!
   > **You MUST click "PUBLISH APP" to switch status to "In Production".**
   > In Production mode, your OAuth refresh token remains valid indefinitely, ensuring your daily GitHub Actions pipeline runs hands-free without recurring re-authorization.

3. **Generate Refresh Token**:
   Run the local helper script:
   ```bash
   python scripts/get_youtube_refresh_token.py
   ```
   Follow the browser prompt to log into your YouTube channel. Copy the returned `YOUTUBE_REFRESH_TOKEN` to your `.env` file.

---

## Running the Pipeline

### Local Dry-Run Mode (Testing)
Generates the script, voiceover, visuals, captions, and renders `output/final_short.mp4` locally without publishing to YouTube or modifying `data/history.json`:

```bash
python -m src.main --dry-run
```

Options:
- `--dry-run`: Render video locally without uploading to YouTube or updating history.
- `--topic "The Pratfall Effect"`: Override automated topic selection with a custom topic.
- `--force-edge-tts`: Force Edge-TTS voiceover synthesis.
- `--keep-temp`: Retain temporary audio clips, still images, and subtitle files in `temp/` for inspection.

### Production Mode (Upload to YouTube)
Generates and uploads directly to your YouTube channel:

```bash
python -m src.main
```

> [!NOTE]
> `data/history.json` is updated **only** after the YouTube API returns a confirmed, valid video ID. If any stage fails, the script exits with code `1` and leaves `data/history.json` untouched.

---

## Daily Hands-Free GitHub Actions Setup

1. Push your repository to GitHub (ensure `.env` is **not** committed; it is ignored in `.gitignore`).
2. In your GitHub repository, go to **Settings > Secrets and variables > Actions**.
3. Under **Repository secrets**, add:
   - `GEMINI_API_KEY`
   - `ELEVENLABS_API_KEY` (optional if using Edge-TTS)
   - `FAL_KEY` (optional if Fal is disabled)
   - `CLOUDFLARE_ACCOUNT_ID`
   - `CLOUDFLARE_API_TOKEN`
   - `YOUTUBE_CLIENT_ID`
   - `YOUTUBE_CLIENT_SECRET`
   - `YOUTUBE_REFRESH_TOKEN`
4. Under **Workflow permissions** (Settings > Actions > General > Workflow permissions):
   - Select **Read and write permissions** so the bot can commit updated `data/history.json` and `data/spend_tracker.json`.
5. The workflow `.github/workflows/daily_short.yml` will trigger automatically every day at **12:00 UTC** (5:30 PM IST), or manually via **Run workflow** in the Actions tab.

---

## Running Tests

Run the test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## License

MIT License. Royalty-free audio placeholders generated by FFmpeg / pure-Python oscillators.
