"""Browser-based web interface for the transcriber project.

A small FastAPI app that accepts **either** an audio recording **or** a picture
of sheet music (PDF / image) and returns a MusicXML score:

* audio   -> :func:`transcriber.transcribe`      (audio   -> MusicXML)
* pdf/img -> :func:`transcriber.omr.recognize`   (optical -> MusicXML)

Run it with::

    transcriber-web                # auto-selects a free port
    transcriber-web --port 8000     # or pick one
    python -m transcriber.web

CORS is enabled so a static frontend (e.g. on GitHub Pages) can call this
backend cross-origin.  The audio and OMR back-ends are imported lazily, so the
app still starts and serves the half whose dependencies are installed.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger(__name__)

# Audio formats we let librosa/ffmpeg attempt to decode.
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
# Sheet-music inputs handled by the OMR pipeline.
OPTICAL_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

app = FastAPI(title="transcriber", description="Audio / sheet-music → MusicXML")

# Allow a separately-hosted static frontend (GitHub Pages, etc.) to call us.
# Restrict with the ALLOWED_ORIGINS env var (comma-separated) in production.
_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>transcriber — audio / sheet music → MusicXML</title>
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
  .opts { display: none; }
  .opts.show { display: block; }
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
  <p class="sub">Upload a recording <em>or</em> a picture of sheet music → get a MusicXML score.</p>
  <form id="form">
    <label for="file">Audio, PDF or image</label>
    <input id="file" name="file" type="file" required
           accept="audio/*,application/pdf,image/*,.wav,.mp3,.flac,.ogg,.m4a,.pdf,.png,.jpg,.jpeg,.tif,.tiff">

    <div id="audioOpts" class="opts">
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
    </div>

    <div id="omrOpts" class="opts">
      <label for="engine">OMR engine</label>
      <select id="engine" name="engine">
        <option value="auto">auto (best installed, else built-in)</option>
        <option value="oemer">oemer (deep learning, needs [omr])</option>
        <option value="primitive">primitive (built-in)</option>
        <option value="ensemble">ensemble (all available)</option>
      </select>
    </div>

    <button id="submit" type="submit">Convert to MusicXML</button>
    <p class="note">Detected type shows the relevant options. The score downloads
       automatically when ready; large files / ML back-ends can take a while.</p>
  </form>
  <div id="status"></div>
<script>
const AUDIO = ['wav','mp3','flac','ogg','m4a','aiff','aif'];
const fileEl = document.getElementById('file');
const audioOpts = document.getElementById('audioOpts');
const omrOpts = document.getElementById('omrOpts');
function ext(name){ const m = /\\.([^.]+)$/.exec(name||''); return m ? m[1].toLowerCase() : ''; }
function refresh(){
  const e = ext(fileEl.files[0] && fileEl.files[0].name);
  const isAudio = AUDIO.includes(e);
  audioOpts.classList.toggle('show', isAudio && !!e);
  omrOpts.classList.toggle('show', !!e && !isAudio);
}
fileEl.addEventListener('change', refresh);

const form = document.getElementById('form');
const status = document.getElementById('status');
const submit = document.getElementById('submit');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileEl.files[0];
  if (!file) return;
  const data = new FormData();
  data.append('file', file);
  data.append('separation', document.getElementById('separation').value);
  data.append('pitch', document.getElementById('pitch').value);
  data.append('drums', document.getElementById('drums').checked ? 'true' : 'false');
  data.append('engine', document.getElementById('engine').value);
  submit.disabled = true;
  status.textContent = 'Converting… this can take a moment.';
  try {
    const resp = await fetch('convert', { method: 'POST', body: data });
    if (!resp.ok) { throw new Error((await resp.text()) || ('HTTP ' + resp.status)); }
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


def _stage_upload(file: UploadFile, data: bytes) -> tuple[Path, Path, str]:
    """Write the upload to a temp dir; return (input_path, out_path, out_name)."""
    suffix = Path(file.filename or "").suffix.lower()
    tmp_dir = Path(tempfile.mkdtemp(prefix="transcriber_"))
    in_path = tmp_dir / f"input{suffix}"
    in_path.write_bytes(data)
    out_name = (Path(file.filename or "score").stem or "score") + ".musicxml"
    return in_path, tmp_dir / out_name, out_name


def _musicxml_response(out_path: Path, out_name: str) -> FileResponse:
    return FileResponse(
        out_path,
        media_type="application/vnd.recordare.musicxml+xml",
        filename=out_name,
    )


@app.post("/convert")
async def convert_endpoint(
    file: UploadFile = File(...),
    separation: str = Form("auto"),
    pitch: str = Form("auto"),
    drums: str = Form("true"),
    engine: str = Form("auto"),
) -> FileResponse:
    """Accept audio OR a sheet-music PDF/image and return a MusicXML score."""
    suffix = Path(file.filename or "").suffix.lower()
    data = await file.read()

    if suffix in AUDIO_SUFFIXES:
        return _convert_audio(file, data, separation, pitch, drums)
    if suffix in OPTICAL_SUFFIXES:
        return _convert_optical(file, data, engine)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type {suffix!r}. Audio: "
        f"{', '.join(sorted(AUDIO_SUFFIXES))}; sheet music: "
        f"{', '.join(sorted(OPTICAL_SUFFIXES))}.",
    )


# Backwards-compatible audio-only endpoint.
@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    separation: str = Form("auto"),
    pitch: str = Form("auto"),
    drums: str = Form("true"),
) -> FileResponse:
    """Accept an uploaded audio file and return a MusicXML score."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. "
            f"Allowed: {', '.join(sorted(AUDIO_SUFFIXES))}",
        )
    return _convert_audio(file, await file.read(), separation, pitch, drums)


def _convert_audio(
    file: UploadFile, data: bytes, separation: str, pitch: str, drums: str
) -> FileResponse:
    from .pipeline import TranscriptionConfig, transcribe

    in_path, out_path, out_name = _stage_upload(file, data)
    config = TranscriptionConfig(
        separation_backend=separation,
        pitch_backend=pitch,
        transcribe_drums=(str(drums).lower() == "true"),
        title=Path(file.filename or "Transcription").stem,
    )
    try:
        transcribe(in_path, out_path, config=config)
    except Exception as exc:  # surface a readable error to the browser
        logger.exception("Audio transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    return _musicxml_response(out_path, out_name)


def _convert_optical(file: UploadFile, data: bytes, engine: str) -> FileResponse:
    from .omr import OMRConfig, recognize

    in_path, out_path, out_name = _stage_upload(file, data)
    config = OMRConfig(engine=engine, title=Path(file.filename or "Optical transcription").stem)
    try:
        recognize(in_path, out_path, config=config)
    except Exception as exc:
        logger.exception("Optical recognition failed")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}") from exc
    return _musicxml_response(out_path, out_name)


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
        help="Port to bind. Defaults to $PORT, else an automatically chosen free port.",
    )
    args = parser.parse_args(argv)

    import uvicorn

    port = args.port or (int(os.environ["PORT"]) if os.environ.get("PORT") else _find_free_port())
    url = f"http://{args.host}:{port}"
    print(f"transcriber web UI running at {url}")
    print("Open that URL in your browser. Press Ctrl+C to stop.")
    uvicorn.run(app, host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
