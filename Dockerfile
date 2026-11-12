# Lightweight Dockerfile for hosting the LUCID web app
# Note: Live interface capture requires tshark/dumpcap capabilities which may not be available on all hosts.
# Hosted demos should use External HTTP ingest and/or included PCAP files.

FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# System deps: tshark for pyshark (optional for HTTP ingest)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tshark ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Add source
COPY . /app

# Expose port (honor $PORT in CMD)
EXPOSE 8000

# Start the web server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
