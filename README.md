# media-transcribe

A TUI application that transcribes **video files**, **audio files**, and **YouTube links**
into clean markdown using the OpenAI Whisper API.

Large media is automatically converted to compact audio, split into chunks small
enough for the Whisper API (25 MB limit), transcribed chunk-by-chunk, and merged
into a single timestamped markdown file saved to a configurable folder.

## Features

- Accepts local video/audio files (mp4, mkv, mp3, wav, m4a, ...) and YouTube URLs
- Automatic audio extraction & compression (64 kbps mono mp3 via ffmpeg)
- Automatic chunking under the Whisper 25 MB limit, timestamps re-aligned after merge
- Timestamped markdown output saved to a configurable directory
- TUI (Textual) with a Settings tab: API key, output folder, model, language, chunk size

## Requirements

- Python 3.10+
- `ffmpeg` / `ffprobe`
- `yt-dlp` (only needed for YouTube links)
- An OpenAI API key

## Install

```bash
git clone https://github.com/HughLyell/media-transcribe.git
cd media-transcribe
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Make sure `ffmpeg` and `yt-dlp` are installed (e.g. `apt install ffmpeg`,
`brew install ffmpeg yt-dlp`, or `nix profile install nixpkgs#ffmpeg nixpkgs#yt-dlp`).

## Run

```bash
python -m media_transcribe
```

1. Open the **Settings** tab, paste your OpenAI API key, choose an output folder,
   and save (stored in `~/.config/media-transcribe/config.json`).
   The `OPENAI_API_KEY` environment variable is also picked up automatically.
2. In the **Transcribe** tab, enter a file path or YouTube link and hit **Transcribe**.
3. The merged markdown file lands in your configured output folder.

## How it works

1. **Source** — local file is used directly; YouTube links are downloaded with `yt-dlp`.
2. **Audio** — media is converted to 16 kHz mono mp3 at 64 kbps (~28 MB/hour).
3. **Chunking** — if the audio exceeds the configured chunk size, ffmpeg splits it
   into fixed-time segments, each safely under the Whisper API limit.
4. **Transcription** — each chunk goes to `whisper-1` with `verbose_json` segments;
   segment timestamps are offset by the chunk's start time.
5. **Merge** — all segments are written to one markdown file:
   `YYYYMMDD-HHMMSS-title-slug.md` in your output folder.

## Config

`~/.config/media-transcribe/config.json`:

```json
{
  "openai_api_key": "sk-...",
  "output_dir": "/home/you/transcriptions",
  "model": "whisper-1",
  "language": "",
  "chunk_mb": 20,
  "include_timestamps": true
}
```
