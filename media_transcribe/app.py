"""Textual TUI for media-transcribe."""

from __future__ import annotations

import tempfile
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Static,
    TabbedContent,
    TabPane,
)

from .config import Config
from .media import MediaError, check_dependencies
from .transcriber import TranscribeError, transcribe_media

CSS = """
#main { padding: 1 2; }
#source-row { height: auto; margin: 1 0; }
#source-input { width: 1fr; }
#transcribe-btn { margin-left: 2; }
#log { height: 1fr; border: solid $primary; margin-top: 1; }
.field { margin: 1 0; }
.field Label { margin-bottom: 0; }
.status-ok { color: $success; }
.status-err { color: $error; }
#save-btn { margin-top: 1; }
"""


class MediaTranscribeApp(App):
    TITLE = "media-transcribe"
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = Config.load()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Transcribe", id="tab-transcribe"):
                with Vertical(id="main"):
                    yield Static(
                        "Enter a path to a video/audio file or paste a YouTube link:",
                    )
                    with Horizontal(id="source-row"):
                        yield Input(
                            placeholder="/path/to/video.mp4  or  https://youtube.com/watch?v=...",
                            id="source-input",
                        )
                        yield Button("Transcribe", id="transcribe-btn", variant="primary")
                    yield Label("", id="status")
                    yield Log(id="log", auto_scroll=True)
            with TabPane("Settings", id="tab-settings"):
                with Vertical(id="main"):
                    yield Label("OpenAI API key", classes="field-label")
                    yield Input(
                        value=self.cfg.openai_api_key,
                        password=True,
                        placeholder="sk-...",
                        id="api-key-input",
                        classes="field",
                    )
                    yield Label("Output folder for markdown transcriptions")
                    yield Input(
                        value=self.cfg.output_dir,
                        id="output-dir-input",
                        classes="field",
                    )
                    yield Label("Whisper model")
                    yield Input(value=self.cfg.model, id="model-input", classes="field")
                    yield Label("Language code (blank = auto-detect, e.g. 'en', 'de')")
                    yield Input(
                        value=self.cfg.language, id="language-input", classes="field"
                    )
                    yield Label("Max chunk size in MB (Whisper limit is 25)")
                    yield Input(
                        value=str(self.cfg.chunk_mb),
                        id="chunk-mb-input",
                        type="integer",
                        classes="field",
                    )
                    yield Checkbox(
                        "Include timestamps in markdown",
                        value=self.cfg.include_timestamps,
                        id="timestamps-checkbox",
                        classes="field",
                    )
                    yield Button("Save settings", id="save-btn", variant="success")
                    yield Label("", id="settings-status")
        yield Footer()

    def on_mount(self) -> None:
        missing = check_dependencies()
        log = self.query_one("#log", Log)
        if missing:
            log.write_line(
                f"WARNING: missing required tools: {', '.join(missing)}. "
                "Install them (ffmpeg, ffprobe, yt-dlp) before transcribing."
            )
        log.write_line(f"Config: {self.cfg_path_hint()}")
        log.write_line(f"Output folder: {self.cfg.output_path}")
        if not self.cfg.openai_api_key:
            log.write_line("No API key set — add one in the Settings tab.")

    def cfg_path_hint(self) -> str:
        from .config import CONFIG_PATH

        return str(CONFIG_PATH)

    def log_line(self, msg: str) -> None:
        self.query_one("#log", Log).write_line(msg)

    def set_status(self, msg: str, ok: bool = True) -> None:
        status = self.query_one("#status", Label)
        status.update(msg)
        status.set_class(not ok, "status-err")
        status.set_class(ok, "status-ok")

    def action_clear_log(self) -> None:
        self.query_one("#log", Log).clear()

    @on(Button.Pressed, "#save-btn")
    def save_settings(self) -> None:
        self.cfg.openai_api_key = self.query_one("#api-key-input", Input).value.strip()
        self.cfg.output_dir = (
            self.query_one("#output-dir-input", Input).value.strip()
            or self.cfg.output_dir
        )
        self.cfg.model = (
            self.query_one("#model-input", Input).value.strip() or "whisper-1"
        )
        self.cfg.language = self.query_one("#language-input", Input).value.strip()
        chunk_raw = self.query_one("#chunk-mb-input", Input).value.strip()
        self.cfg.chunk_mb = int(chunk_raw) if chunk_raw.isdigit() else self.cfg.chunk_mb
        self.cfg.chunk_mb = min(max(self.cfg.chunk_mb, 1), 24)
        self.cfg.include_timestamps = self.query_one(
            "#timestamps-checkbox", Checkbox
        ).value
        self.cfg.save()
        self.query_one("#settings-status", Label).update("Settings saved.")

    @on(Button.Pressed, "#transcribe-btn")
    @on(Input.Submitted, "#source-input")
    def start_transcription(self) -> None:
        if self._busy:
            self.set_status("Already transcribing — please wait.", ok=False)
            return
        source = self.query_one("#source-input", Input).value.strip()
        if not source:
            self.set_status("Please enter a file path or YouTube link.", ok=False)
            return
        self.save_settings()  # persist latest settings before running
        self._busy = True
        self.query_one("#transcribe-btn", Button).disabled = True
        self.set_status("Working...")
        self.log_line(f"--- New job: {source}")
        self.run_transcription(source)

    @work(thread=True)
    def run_transcription(self, source: str) -> None:
        log = self.log_line
        try:
            with tempfile.TemporaryDirectory(prefix="media-transcribe-") as tmp:
                out_path = transcribe_media(source, self.cfg, Path(tmp), log)
        except (MediaError, TranscribeError) as exc:
            self.call_from_thread(self._finish, None, str(exc))
        except Exception as exc:  # noqa: BLE001 - surface API errors in the TUI
            self.call_from_thread(self._finish, None, f"Unexpected error: {exc}")
        else:
            self.call_from_thread(self._finish, out_path, None)

    def _finish(self, out_path: Path | None, error: str | None) -> None:
        self._busy = False
        self.query_one("#transcribe-btn", Button).disabled = False
        if error:
            self.log_line(f"ERROR: {error}")
            self.set_status("Failed — see log.", ok=False)
        else:
            self.set_status(f"Done: {out_path}")


def main() -> None:
    MediaTranscribeApp().run()


if __name__ == "__main__":
    main()
