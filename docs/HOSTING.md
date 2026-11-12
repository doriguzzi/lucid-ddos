# Free Hosting Options for LUCID Web App

This app is a Python FastAPI server with a WebSocket and a small HTML/JS dashboard. It can be hosted for free in a few ways. Important constraints:

- Live interface capture requires `tshark`/`dumpcap` with OS capabilities; most hosted platforms don’t allow this. Use External HTTP ingest or PCAP replay in hosted demos.
- TensorFlow is heavy; free tiers may have low RAM/CPU. Keep models small; CPU inference is fine.

## Option A: Render (free web service)

Render supports Docker-based web services with a free tier suitable for demos.

1. Fork this repo to your GitHub account.
2. Ensure the `Dockerfile` and `render.yaml` at repo root are present (they are in this repo).
3. Create a new Web Service on <https://render.com>:
   - Connect your forked repo
   - Render will detect `render.yaml` (Infrastructure as Code) automatically
   - Choose the Free plan when prompted
4. Once deployed, open the public URL. The dashboard should load.

Without Docker on Render (Native Python):

- Add a `runtime.txt` at repo root with: `python-3.10.13` (avoids Python 3.13 incompatibilities with TensorFlow)
- Build Command: `python -m pip install -U pip setuptools wheel && pip install -r requirements-hosted.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- (Optional) Add `Procfile` with the same start command; some platforms detect this automatically.

Note: `requirements-hosted.txt` pins FastAPI to 0.95.x and Pydantic 1.10.x to satisfy TensorFlow 2.12’s dependency on typing-extensions<4.6. If your code uses Pydantic v2 features, tell us and we’ll bump TF and adjust pins instead.

Notes:

- Prefer “External HTTP ingest” mode in production demos. PCAP replay can work, but large PCAP files may exceed free disk/time limits.
- Live capture over an interface is typically not possible in hosted containers.

## Option B: Hugging Face Spaces (Docker)

Hugging Face Spaces can host arbitrary Docker apps for free (with usage limits):

1. Create a new Space → Type: Docker.
2. Push this repo (or minimal subset) with the provided `Dockerfile`.
3. The Space will build the container and expose a URL.

Limitations: Long cold starts, resource caps, and no raw packet capture. Use HTTP ingest/PCAP.

Without Docker alternative: Spaces primarily expects Gradio/Streamlit or Docker for custom FastAPI apps. If you prefer no Docker, consider Render or Railway instead.

## Option C: Local backend + public URL (free tunnel)

Run the server locally (full features including live capture) and expose it with a secure tunnel:

- Cloudflare Tunnel (free):
  1. Install `cloudflared`
  2. Run `cloudflared tunnel --url http://127.0.0.1:8000`
  3. Share the generated public URL while your server runs locally

- (Alternative) Localhost.run or other free tunnels can work similarly.

This gives you full functionality (including interface capture) without hosting restrictions.

Tip: You can also use Railway (no Docker) — connect your repo, set Build Command to `pip install -r requirements-hosted.txt` and Start Command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. WebSockets are supported.

## Platform caveats

- WebSockets: Render supports them out of the box. Spaces also work with the provided Dockerfile.
- Memory: Use CPU TensorFlow and a compact model; avoid training on free instances.
- File access: Hosted containers can access files in the repo (e.g., small PCAPs), but uploads/storage may be limited.

## Minimal steps for a quick free demo (Render)

- Use External HTTP ingest:
  - Deploy on Render (Option A)
  - Visit `/` and click Start with Source = External HTTP ingest
  - In another terminal, POST fragments to `/api/ingest` (see README_WINDOWS.md or README.md for examples)

That’s it—you’ll have a live URL to your dashboard.
