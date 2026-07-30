# --- Base image: official slim Python 3.12 ---
# "slim" = minimal OS, smaller image, faster pulls. Pinned to 3.12 (not 3.11)
# because requirements.txt pins numpy==2.5.1, which requires Python >=3.12 -
# discovered via a failed build attempt on 3.11. Pinned explicitly rather
# than "latest" so the build stays reproducible regardless of when it's rebuilt.
FROM python:3.12-slim

# --- Working directory inside the container ---
# All following commands (COPY, RUN, CMD) run relative to this path.
WORKDIR /app

# --- System dependencies ---
# torch / sentence-transformers sometimes need build tools to install cleanly.
# Installed first (before requirements.txt) since this layer changes rarely.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies FIRST, before copying app code ---
# This is the key caching trick: Docker only re-runs this layer if
# requirements.txt itself changes. If you only edit graph/nodes.py later,
# Docker reuses this cached layer instead of reinstalling torch etc. again
# (which is slow) - massive rebuild time savings.
COPY requirements.txt .

# torch is installed FIRST and SEPARATELY from the CPU-only wheel index.
# Plain "pip install torch" defaults to the full CUDA/GPU build, which pulls
# several GB of Nvidia libraries (nvidia-cuda-*, nvidia-cufft, etc.) that are
# useless here - our deployment host has no GPU. The CPU-only index at
# download.pytorch.org/whl/cpu skips all of that, cutting the image by GBs
# and the build time drastically.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Now install everything else normally. torch==2.12.1 is already satisfied
# above, so pip won't try to reinstall the GPU version for it here.
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download models into the image so the container never needs
# network access to Hugging Face at runtime (avoids rate-limit failures
# and speeds up cold starts significantly).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# --- Now copy the actual application code ---
# This layer changes every time you edit code, so it's placed last -
# keeps the expensive dependency layer above cached and untouched.
COPY . .

# --- Document which port the app listens on ---
# This is metadata only (doesn't actually publish the port) - actual
# port mapping happens with `docker run -p` later.

# --- The command that runs when the container starts ---
# --host 0.0.0.0 is mandatory in Docker: 127.0.0.1 (localhost) inside a
# container is only reachable from INSIDE that container. 0.0.0.0 means
# "listen on all network interfaces," which lets Docker's port mapping
# actually forward traffic in from outside.

EXPOSE 8080
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}