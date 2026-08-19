"""Whisper API transcription and markdown assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .config import Config
from .media import prepare_audio, probe_duration


class TranscribeError(RuntimeError):
    pass


@dataclass
class Segment:
    start: float  # seconds, absolute in source media
    text: str


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\- ]", "", title.lower()).strip()
    return re.sub(r"[\s_]+", "-", slug)[:80] or "transcription"


def transcribe_chunks(
    client: OpenAI,
    chunks: list[tuple[Path, float]],
    model: str,
    language: str,
    log=print,
) -> list[Segment]:
    """Transcribe each chunk with Whisper, shifting timestamps to absolute time."""
    segments: list[Segment] = []
    total = len(chunks)
    for i, (chunk_path, offset) in enumerate(chunks, 1):
        log(f"Transcribing chunk {i}/{total} ({chunk_path.name})...")
        kwargs: dict = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language
        with open(chunk_path, "rb") as fh:
            result = client.audio.transcriptions.create(file=fh, **kwargs)

        raw = getattr(result, "segments", None)
        if raw:
            for seg in raw:
                start = seg.start if hasattr(seg, "start") else seg["start"]
                text = seg.text if hasattr(seg, "text") else seg["text"]
                segments.append(Segment(start=offset + start, text=text.strip()))
        else:
            text = getattr(result, "text", "") or ""
            if text.strip():
                segments.append(Segment(start=offset, text=text.strip()))
        log(f"Chunk {i}/{total} done ({len(segments)} segments so far)")
    return segments


def render_markdown(
    title: str,
    source: str,
    segments: list[Segment],
    duration: float | None,
    model: str,
    include_timestamps: bool,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- **Source:** {source}",
        f"- **Transcribed:** {datetime.now():%Y-%m-%d %H:%M}",
        f"- **Model:** `{model}`",
    ]
    if duration is not None:
        lines.append(f"- **Duration:** {format_timestamp(duration)}")
    lines += ["", "---", ""]

    if include_timestamps:
        lines += [f"**[{format_timestamp(s.start)}]** {s.text}" for s in segments]
    else:
        # Reflow into paragraphs, breaking on sentence-ish pauses.
        para: list[str] = []
        for s in segments:
            para.append(s.text)
            if s.text.endswith((".", "?", "!")) and sum(len(t) for t in para) > 400:
                lines.append(" ".join(para))
                lines.append("")
                para = []
        if para:
            lines.append(" ".join(para))

    lines.append("")
    return "\n".join(lines)


def transcribe_media(source: str, cfg: Config, work_dir: Path, log=print) -> Path:
    """End-to-end: source -> chunks -> Whisper -> merged markdown file.

    Returns the path of the written markdown file.
    """
    if not cfg.openai_api_key:
        raise TranscribeError(
            "No OpenAI API key configured. Add one in Settings or set OPENAI_API_KEY."
        )

    chunks, title = prepare_audio(source, work_dir, cfg.chunk_bytes, log)
    try:
        duration = probe_duration(chunks[0][0]) if len(chunks) == 1 else None
        client = OpenAI(api_key=cfg.openai_api_key)
        segments = transcribe_chunks(client, chunks, cfg.model, cfg.language, log)
        if not segments:
            raise TranscribeError("Whisper returned no transcription text")

        body = render_markdown(
            title, source, segments, duration, cfg.model, cfg.include_timestamps
        )

        out_dir = cfg.output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"{stamp}-{slugify(title)}.md"
        out_path.write_text(body)
        log(f"Saved transcription -> {out_path}")
        return out_path
    finally:
        # Clean up chunk files but keep the converted audio out of the way.
        for chunk_path, _ in chunks:
            if chunk_path.name.startswith("chunk_"):
                chunk_path.unlink(missing_ok=True)
