#!/usr/bin/env bash
set -euo pipefail

# Install tshark non-interactively for PyShark
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tshark

# Ensure we install into the SAME Python env that will run the app
# Prefer Render's runtime venv if present
TARGET_PY="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
if [ -z "${TARGET_PY:-}" ] || [ ! -x "$TARGET_PY" ]; then
	if [ -x "/opt/render/project/src/.venv/bin/python" ]; then
		TARGET_PY="/opt/render/project/src/.venv/bin/python"
	else
		TARGET_PY="$(command -v python)"
	fi
fi
echo "Using Python: $TARGET_PY"
"$TARGET_PY" -V || true
"$TARGET_PY" -c "import sys; print('python exe:', sys.executable)" || true

# Upgrade pip tooling inside target Python and install deps (hosted pins)
"$TARGET_PY" -m pip install --upgrade pip setuptools wheel
"$TARGET_PY" -m pip install -r requirements-hosted.txt

# Verify uvicorn installed (module form) and show pip context
"$TARGET_PY" -m pip --version || true
"$TARGET_PY" -c "import uvicorn, sys; print('uvicorn module OK, version:', getattr(uvicorn, '__version__', 'unknown')); print('site:', sys.path[:3])" || true

# Sanity check (optional)
which tshark || true
(tshark -v || true)
echo "Build script completed." 