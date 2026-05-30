"""Browser-based web interface for the transcriber pipeline.

A tiny FastAPI app that lets you upload an audio file, runs the
audio-to-MusicXML pipeline and returns the resulting score as a download.

Run it with::

    transcriber-web                # auto-selects a free port
    transcriber-web --port 8000     # or pick one
    python -m transcriber.web

The heavy ML back-ends are used automatically when installed (see the
``[full]`` extra); otherwise the librosa fallbacks are used.
"""

from __future__ import annotations

import argparse
import logging
import socket
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .pipeline import TranscriptionConfig, transcribe

logger = logging.getLogger(__name__)

# Audio formats we let librosa/ffmpeg attempt to decode.
ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

app = FastAPI(title="transcriber", description="Audio → MusicXML")


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>transcriber — audio → MusicXML</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 640px;
         margin: 3rem auto; padding: 0 1rem; line-height: 1.5; }
  h1 { margin-bottom: 0.2rem; }
  .sub { color: #888; margin-top: 0; }
  form { border: 1px solid #8884; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; }
  label { display: block; margin: 0.75rem 0 0.25rem; font-weight: 600; }
  input, select { width: 100%; padding: 0.5rem; box-sizing: border-box;
                  border-radius: 8px; border: 1px solid #8886; background: transparent; color: inherit; }
  .row { display: flex; gap: 1rem; }
  .row > div { flex: 1; }
  button { margin-top: 1.25rem; padding: 0.7rem 1.2rem; border: 0; border-radius: 8px;
           background: #4f46e5; color: white; font-size: 1rem; cursor: pointer; width: 100%; }
  button:disabled { opacity: 0.6; cursor: progress; }
  .note { color: #888; font-size: 0.85rem; }
  #status { margin-top: 1rem; font-weight: 600; }
</style>
</head>
<body>
  <h1>transcriber</h1>
  <p class="sub">Upload a recording → get a MusicXML score.</p>
  <form id="form">
    <label for="file">Audio file</label>
    <input id="file" name="file" type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a" required>
    <div class="row">
      <div>
        <label for="separation">Separation</label>
        <select id="separation" name="separation">
          <option value="auto">auto</option>
          <option value="demucs">demucs (needs [full])</option>
          <option value="hpss">hpss (fallback)</option>
        </select>
      </div>
      <div>
        <label for="pitch">Pitch</label>
        <select id="pitch" name="pitch">
          <option value="auto">auto</option>
          <option value="basic-pitch">basic-pitch (needs [full])</option>
          <option value="pyin">pyin (fallback)</option>
        </select>
      </div>
    </div>
    <label><input type="checkbox" id="drums" name="drums" checked style="width:auto"> Transcribe drums</label>
    <button id="submit" type="submit">Transcribe</button>
    <p class="note">Large files and the ML back-ends can take a while. The score
       downloads automatically when ready.</p>
  </form>
  <div id="status"></div>
<script>
const form = document.getElementById('form');
const status = document.getElementById('status');
const submit = document.getElementById('submit');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = new FormData();
  const file = document.getElementById('file').files[0];
  if (!file) return;
  data.append('file', file);
  data.append('separation', document.getElementById('separation').value);
  data.append('pitch', document.getElementById('pitch').value);
  data.append('drums', document.getElementById('drums').checked ? 'true' : 'false');
  submit.disabled = true;
  status.textContent = 'Transcribing… this can take a moment.';
  try {
    const resp = await fetch('/transcribe', { method: 'POST', body: data });
    if (!resp.ok) {
      const msg = await resp.text();
      throw new Error(msg || ('HTTP ' + resp.status));
    }
    const blob = await resp.blob();
    const name = (file.name.replace(/\\.[^.]+$/, '') || 'score') + '.musicxml';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    URL.revokeObjectURL(url);
    status.textContent = 'Done — downloaded ' + name;
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  } finally {
    submit.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    separation: str = Form("auto"),
    pitch: str = Form("auto"),
    drums: str = Form("true"),
) -> FileResponse:
    """Accept an uploaded audio file and return a MusicXML score."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    # Stage upload and output in a temp dir; FileResponse streams the result.
    tmp_dir = Path(tempfile.mkdtemp(prefix="transcriber_"))
    audio_path = tmp_dir / f"input{suffix}"
    audio_path.write_bytes(await file.read())

    out_name = (Path(file.filename or "score").stem or "score") + ".musicxml"
    out_path = tmp_dir / out_name

    config = TranscriptionConfig(
        separation_backend=separation,
        pitch_backend=pitch,
        transcribe_drums=(str(drums).lower() == "true"),
        title=Path(file.filename or "Transcription").stem,
    )
    try:
        transcribe(audio_path, out_path, config=config)
    except Exception as exc:  # surface a readable error to the browser
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    return FileResponse(
        out_path,
        media_type="application/vnd.recordare.musicxml+xml",
        filename=out_name,
    )


def _find_free_port() -> int:
    """Ask the OS for an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcriber-web",
        description="Run the transcriber web interface.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind. Defaults to an automatically chosen free port.",
    )
    args = parser.parse_args(argv)

    import uvicorn

    port = args.port or _find_free_port()
    url = f"http://{args.host}:{port}"
    print(f"transcriber web UI running at {url}")
    print("Open that URL in your browser. Press Ctrl+C to stop.")
    uvicorn.run(app, host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
