"""Media handling: YouTube download, audio extraction, and chunking.

Relies on external tools: ffmpeg, ffprobe, and yt-dlp.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg",
}

YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/(watch\?|shorts/|live/)|youtu\.be/)\S+"
)

# Target for audio conversion: small, speech-friendly, well supported by Whisper.
TARGET_BITRATE_KBPS = 64
TARGET_SAMPLE_RATE = 16000


class MediaError(RuntimeError):
    pass


def check_dependencies() -> list[str]:
    """Return a list of missing external tools."""
    return [tool for tool in ("ffmpeg", "ffprobe", "yt-dlp") if shutil.which(tool) is None]


def is_youtube_url(source: str) -> bool:
    return bool(YOUTUBE_RE.match(source.strip()))


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"{' '.join(cmd[:2])} failed:\n{proc.stderr.strip()[-2000:]}")
    return proc


def download_youtube(url: str, work_dir: Path, log=print) -> tuple[Path, str]:
    """Download a YouTube video's audio track. Returns (audio_path, title)."""
    log(f"Downloading audio from YouTube: {url}")
    out_tmpl = str(work_dir / "%(title).80s.%(ext)s")
    _run([
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "--no-playlist",
        "--print", "after_move:filepath",
        "-o", out_tmpl,
        url,
    ])
    files = sorted(work_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise MediaError("yt-dlp finished but produced no audio file")
    path = files[-1]
    title = path.stem
    log(f"Downloaded: {title}")
    return path, title


def probe_duration(path: Path) -> float:
    """Return media duration in seconds."""
    proc = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise MediaError(f"Could not determine duration of {path}") from exc


def to_audio(source: Path, work_dir: Path, log=print) -> Path:
    """Convert any media file to a compact mono mp3 for Whisper."""
    out = work_dir / "audio.mp3"
    log(f"Extracting/converting audio -> mp3 {TARGET_BITRATE_KBPS}kbps mono...")
    _run([
        "ffmpeg", "-y", "-i", str(source),
        "-vn", "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-b:a", f"{TARGET_BITRATE_KBPS}k",
        str(out),
    ])
    return out


def split_into_chunks(
    audio: Path, work_dir: Path, chunk_bytes: int, log=print
) -> list[tuple[Path, float]]:
    """Split audio into chunks under chunk_bytes.

    Returns a list of (chunk_path, start_offset_seconds) in order.
    At 64 kbps mono, one second of audio ~= 8 KB, so a byte budget maps
    directly to a segment duration.
    """
    size = audio.stat().st_size
    if size <= chunk_bytes:
        log(f"Audio is {size / 1e6:.1f} MB — no splitting needed")
        return [(audio, 0.0)]

    bytes_per_sec = TARGET_BITRATE_KBPS * 1024 / 8
    segment_time = max(60, int(chunk_bytes / bytes_per_sec) - 5)  # safety margin
    n_chunks = size // chunk_bytes + 1
    log(
        f"Audio is {size / 1e6:.1f} MB — splitting into ~{n_chunks} "
        f"chunks of {segment_time // 60}m{segment_time % 60:02d}s"
    )

    pattern = str(work_dir / "chunk_%05d.mp3")
    _run([
        "ffmpeg", "-y", "-i", str(audio),
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-c", "copy",
        pattern,
    ])
    chunks = sorted(work_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise MediaError("ffmpeg segmentation produced no chunks")
    log(f"Created {len(chunks)} chunks")
    return [(c, i * float(segment_time)) for i, c in enumerate(chunks)]


def prepare_audio(source: str, work_dir: Path, chunk_bytes: int, log=print):
    """Full pipeline: obtain audio from source and split into Whisper-ready chunks.

    Returns (chunks_with_offsets, title).
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    if is_youtube_url(source):
        media, title = download_youtube(source.strip(), work_dir, log)
    else:
        media = Path(source).expanduser()
        if not media.is_file():
            raise MediaError(f"File not found: {media}")
        title = media.stem
        log(f"Using local file: {media.name}")

    audio = to_audio(media, work_dir, log)
    chunks = split_into_chunks(audio, work_dir, chunk_bytes, log)
    return chunks, title
