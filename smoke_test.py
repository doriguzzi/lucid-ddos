#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    p = argparse.ArgumentParser(description="Simple smoke test for LUCID web service")
    p.add_argument("--base-url", required=True, help="Base URL, e.g., http://localhost:8000")
    p.add_argument("--source", required=True, help="PCAP path relative to server working dir (or absolute)")
    p.add_argument("--model_path", default=None, help="Optional model path; server will autodiscover if omitted")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=60, help="Seconds to wait for samples")
    args = p.parse_args()

    base = args.base_url.rstrip("/")

    # Stop any previous run
    try:
        http_post(f"{base}/api/stop", {})
    except Exception:
        pass

    # Start detection
    start_payload = {
        "source": args.source,
        "threshold": args.threshold,
    }
    if args.model_path:
        start_payload["model_path"] = args.model_path

    try:
        resp = http_post(f"{base}/api/start", start_payload)
        if not resp.get("ok", False):
            print("Start response not ok:", resp, file=sys.stderr)
            return 2
    except urllib.error.HTTPError as e:
        print("Start failed:", e.read().decode("utf-8"), file=sys.stderr)
        return 2
    except Exception as e:
        print("Start failed:", e, file=sys.stderr)
        return 2

    # Poll status until samples observed or timeout
    deadline = time.time() + args.timeout
    observed_samples = False
    last_status = None
    try:
        while time.time() < deadline:
            st = http_get(f"{base}/api/status")
            last_status = st
            status = st.get("status")
            pos_rate = st.get("positive_rate")
            if status == "error":
                print("Service error:", st.get("last_error"), file=sys.stderr)
                return 3
            # history is a list of window payloads; samples key per window indicates count
            # we succeed once any window reports samples > 0
            # fetch history for detailed confirmation if needed
            hist = http_get(f"{base}/api/history")
            if isinstance(hist, list) and any((isinstance(h, dict) and int(h.get("samples", 0)) > 0) for h in hist):
                observed_samples = True
                break
            time.sleep(2)
    finally:
        try:
            http_post(f"{base}/api/stop", {})
        except Exception:
            pass

    if not observed_samples:
        print("Timeout waiting for samples. Last status:")
        print(json.dumps(last_status or {}, indent=2))
        return 4

    print("Smoke test passed: samples observed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
