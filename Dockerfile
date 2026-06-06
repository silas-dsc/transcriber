# Backend container for the transcriber web API (audio / sheet music -> MusicXML).
#
# Runs the FastAPI app from transcriber.web.  Listens on $PORT (default 7860,
# which is Hugging Face Spaces' default; Render/Cloud Run/Fly set $PORT).
#
# Build:  docker build -t transcriber .
# Run:    docker run -p 7860:7860 transcriber
#
# This installs the core pipeline (audio fallbacks + built-in OMR) plus the web
# server.  The heavy ML back-ends are optional:
#   * OMR deep-learning engine:   pip install oemer        (set INSTALL_OEMER=1)
#   * audio ML back-ends:         pip install ".[full]"    (large; Demucs/torch)
FROM python:3.11-slim

# ffmpeg: decode mp3/m4a for the audio pipeline.  libsndfile1: soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ARG INSTALL_OEMER=0
RUN pip install --no-cache-dir ".[web]" \
    && if [ "$INSTALL_OEMER" = "1" ]; then pip install --no-cache-dir oemer; fi

# Restrict browser origins that may call the API by setting ALLOWED_ORIGINS,
# e.g. ALLOWED_ORIGINS="https://<user>.github.io".  Defaults to "*".
ENV ALLOWED_ORIGINS=*
EXPOSE 7860

CMD ["sh", "-c", "uvicorn transcriber.web:app --host 0.0.0.0 --port ${PORT:-7860}"]
