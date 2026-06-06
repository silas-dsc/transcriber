# Deploying transcriber (free hosting)

Two pieces:

- **Frontend** — a static page (`frontend/index.html`) → **GitHub Pages** (free).
- **Backend** — the FastAPI app (`transcriber.web`) → a free Python host.
  **Hugging Face Spaces** is the easiest free option for this kind of
  ML-flavoured Python service; Render's free web service also works.

```
 browser ──▶ GitHub Pages (static frontend)
                  │  fetch(`${BACKEND}/convert`)   (CORS)
                  ▼
            Backend API (Hugging Face Space / Render / Fly / Cloud Run)
                  │  audio  → transcribe()      → MusicXML
                  └  pdf/img → omr.recognize()  → MusicXML
```

---

## 1. Get the code onto your Mac and into your own GitHub account

I can't reach your laptop or your personal GitHub account from here, so run
these locally. They clone the branch with all this work and push it to a brand
new repo under your account (shown here as the user `silasp`).

```bash
# 1a. Clone the current work into the path you asked for.
cd /Users/silaspalmer/Websites
git clone https://github.com/silas-dsc/transcriber.git transcriber
cd transcriber
git checkout claude/omr-system-improvement-HBDJ1   # the branch with all this

# 1b. Create an EMPTY repo on your own account first:
#     github.com → New repository → owner: silasp, name: transcriber,
#     do NOT initialise with a README. Then:

# 1c. Point "origin" at your new repo and push.
git remote rename origin upstream                 # keep the old one as 'upstream'
git remote add origin https://github.com/silasp/transcriber.git
git push -u origin claude/omr-system-improvement-HBDJ1:main   # push work as 'main'
```

> Authentication: if `git push` rejects you, create a Personal Access Token
> (github.com → Settings → Developer settings → Tokens) and use it as the
> password, or set up the `gh` CLI (`brew install gh && gh auth login`).
> With `gh` you can even create the repo in one step:
> `gh repo create silasp/transcriber --public --source=. --push`.

After this, `silasp/transcriber` on GitHub has everything.

---

## 2. Deploy the backend (free)

### Option A — Hugging Face Spaces (recommended, free)

1. Create an account at <https://huggingface.co>.
2. **New Space** → SDK: **Docker** → name it (e.g. `transcriber`).
3. Push this repo to the Space (it already has a `Dockerfile`):
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/transcriber
   git push space main
   ```
   (or connect the GitHub repo to the Space in the Space settings).
4. The Space builds the `Dockerfile` and serves the API. Note its URL, e.g.
   `https://<you>-transcriber.hf.space`. Test it: open `…/health` → `{"status":"ok"}`.
5. Recommended: set the Space secret/variable `ALLOWED_ORIGINS` to your Pages
   origin (e.g. `https://silasp.github.io`) so only your frontend can call it.

The default Space CPU tier is free and runs the built-in OMR + audio fallbacks.
To add the deep-learning OMR engine, set the build arg `INSTALL_OEMER=1` (or
`pip install oemer` in the Dockerfile) — it needs more memory/build time.

### Option B — Render (free web service)

1. <https://render.com> → **New → Web Service** → connect `silasp/transcriber`.
2. Environment: **Docker** (it uses the `Dockerfile`). Render sets `$PORT`
   automatically; the app already reads it.
3. Set env var `ALLOWED_ORIGINS=https://silasp.github.io`.
4. Free instances sleep when idle and cold-start on the next request (slow
   first hit) — fine for a demo.

### Other options

- **Fly.io** / **Google Cloud Run** — both have free allowances and run the
  same `Dockerfile` (they set `$PORT`). Cloud Run scales to zero (no idle cost).
- **Railway** — easy, small free credit.

> **Heavy ML caveat:** Demucs/basic-pitch (audio) and large OMR models pull in
> torch/onnxruntime and big weights — usually too big for free tiers. The
> default deploy intentionally uses the light audio fallbacks + built-in OMR,
> so it fits. Add the heavy extras only on a paid tier.

---

## 3. Deploy the frontend on GitHub Pages (free)

This repo includes `.github/workflows/pages.yml`, which publishes
`frontend/` to Pages on every push.

1. On `silasp/transcriber`: **Settings → Pages → Build and deployment →
   Source: GitHub Actions**.
2. Push to `main` (the workflow runs and deploys). Your site appears at
   `https://silasp.github.io/transcriber/`.
3. Open it, paste your backend URL (from step 2) into the **Backend URL** field
   (it's saved in your browser), and convert a file.
   - You can also share a pre-filled link:
     `https://silasp.github.io/transcriber/?api=https://<you>-transcriber.hf.space`

> Prefer zero workflow? Copy `frontend/index.html` to `docs/index.html`,
> commit, and set **Pages → Source: Deploy from a branch → /docs**.

---

## 4. Verify end to end

1. Backend health: `curl https://<backend>/health` → `{"status":"ok"}`.
2. Open the Pages site, set the Backend URL.
3. Upload a short audio clip → downloads a `.musicxml`.
4. Upload a sheet-music PDF or image → downloads a `.musicxml`.

If the browser console shows a CORS error, make sure `ALLOWED_ORIGINS` on the
backend includes your exact Pages origin (scheme + host, no trailing slash).
