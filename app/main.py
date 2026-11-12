import os
import sys
import time
import threading
import logging
import asyncio
import queue
import shutil
import subprocess
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure we can import project modules (lucid_*.py at repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pyshark
from tensorflow.keras.models import load_model

from util_functions import static_min_max, normalize_and_padding, count_packets_in_dataset
from lucid_dataset_parser import process_live_traffic, parse_labels, dataset_to_list_of_fragments


class StartConfig(BaseModel):
    source: str  # network interface name or path to pcap file
    # If omitted, the server will use env LUCID_MODEL_PATH or auto-discover a .h5 under ./output
    model_path: Optional[str] = None
    dataset_type: Optional[str] = None  # DOS2017 | DOS2018 | DOS2019 | SYN2020 (optional for accuracy calc)
    attack_net: Optional[str] = None    # optional for accuracy calc on custom traffic
    victim_net: Optional[str] = None    # optional for accuracy calc on custom traffic
    threshold: float = 0.5              # threshold on fraction of predicted ddos windows to trigger alert
    source_kind: Optional[str] = None   # iface | pcap | http (external push)
    # Optional: force ground-truth labeling to help compute KPIs when labels are unavailable
    # one of: 'all_attack', 'all_benign'
    ground_truth_override: Optional[str] = None
    # PCAP playback options (optional)
    pcap_realtime: Optional[bool] = None  # if true, throttle PCAP windows to real time (sleep time_window between windows)
    pcap_loop: Optional[bool] = None      # if true, loop the PCAP file when it reaches EOF


class IngestPayload(BaseModel):
    # Each fragment is a 2D list (packets x features) as expected by normalize_and_padding
    fragments: List[List[List[float]]]
    # Optional metadata aligned to fragments for UI/mitigation metrics
    src_ips: Optional[List[Optional[str]]] = None
    dest_ports: Optional[List[Optional[int]]] = None
    ts: Optional[float] = None
    # Optional ground-truth labels per fragment (0/1), used to compute TPR/FPR/TTD in HTTP mode
    labels: Optional[List[Optional[int]]] = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        failed: List[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                failed.append(connection)
        for ws in failed:
            self.disconnect(ws)


class DetectorService:
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status: str = "idle"
        self._last_error: Optional[str] = None
        self._history: List[Dict[str, Any]] = []  # recent metrics and alerts

        # dynamic runtime state
        self.cap: Optional[pyshark.LiveCapture] = None
        self.cap_file: Optional[pyshark.FileCapture] = None
        self.model = None
        self.model_info: Dict[str, Any] = {}
        self.labels = None
        self.time_window: int = 10
        self.max_flow_len: int = 10
        self.threshold: float = 0.5

        # mitigation simulation
        self.blocked_sources = set()

        # predictive analytics (simple EWMA of ddos_fraction)
        self._ewma_ddos = None
        self._ewma_alpha = 0.5
        # reference to the asyncio event loop for cross-thread broadcasts
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        # external ingest buffer and source kind
        self._ingest: Optional["queue.Queue"] = None
        self._source_kind: str = "pcap"
        # PCAP playback options
        self._pcap_path: Optional[str] = None
        self._pcap_realtime: bool = False
        self._pcap_loop: bool = False
        # auto-mitigation settings
        self._auto_block_limit = 20  # cap number of sources auto-blocked per window
        self.hysteresis_windows = 2  # require N consecutive alert windows before acting
        self.cooldown_seconds = 120  # do not re-act on same source within this window (seconds)
        self.allowlist = set()       # sources that should never be blocked
        self._consecutive_alerts = 0
        # mitigation state per source: {ip: {stage, last_action, cooldown_until, pos_count}}
        self._mitigation: Dict[str, Dict[str, Any]] = {}

        # accuracy-at-scale metrics (optional if labels available)
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        self._attack_active = False
        self._attack_start_ts: Optional[float] = None
        self._first_alert_ts: Optional[float] = None
        self._ttds: List[float] = []
        # window-based indexing for robust TTD
        self._window_index = 0
        self._attack_start_window = None

        # downtime prevention accounting
        self._prevented_downtime_min = 0.0
        # KPI configuration
        self._gt_override = None
        self._kpi_source = "none"  # dataset | http-labels | override | none
        # cumulative prediction accounting
        self._pos_total = 0
        self._samples_total = 0

    def _autodiscover_model_path(self) -> Optional[str]:
        """Try to find a model .h5 if none provided, preferring ./output then repo root.
        Returns absolute path or None if not found.
        """
        search_dirs = [
            os.path.join(REPO_ROOT, "output"),
            REPO_ROOT,
        ]
        candidates: List[str] = []
        for d in search_dirs:
            try:
                for name in os.listdir(d):
                    if name.lower().endswith(".h5"):
                        candidates.append(os.path.join(d, name))
            except Exception:
                continue
        if not candidates:
            return None
        # pick most recently modified
        try:
            best = max(candidates, key=lambda p: os.path.getmtime(p))
            return best
        except Exception:
            return candidates[0]

    # Diagnostics helpers
    @staticmethod
    def _tshark_path() -> Optional[str]:
        return shutil.which("tshark")

    @staticmethod
    def list_interfaces() -> List[Dict[str, str]]:
        """Return list of capture interfaces using `tshark -D` if available."""
        path = DetectorService._tshark_path()
        if not path:
            return []
        try:
            proc = subprocess.run([path, "-D"], capture_output=True, text=True, timeout=5)
            out = proc.stdout or ""
        except Exception:
            return []
        result: List[Dict[str, str]] = []
        for line in out.splitlines():
            # Format: "1. eth0 (Linux cooked capture)"
            line = line.strip()
            if not line or "." not in line:
                continue
            try:
                idx_part, rest = line.split(".", 1)
                name_desc = rest.strip()
                # name is token until space or (desc)
                name = name_desc.split(" ")[0]
                desc = name_desc[len(name):].strip()
                result.append({"name": name, "desc": desc})
            except Exception:
                continue
        return result

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._async_loop = loop

    def status(self) -> Dict[str, Any]:
        # compute live accuracy metrics if any labels observed
        tpr = (self._tp / (self._tp + self._fn)) if (self._tp + self._fn) > 0 else None
        fpr = (self._fp / (self._fp + self._tn)) if (self._fp + self._tn) > 0 else None
        ttd = (sum(self._ttds) / len(self._ttds)) if self._ttds else None
        pos_rate = (self._pos_total / self._samples_total) if self._samples_total > 0 else None
        return {
            "status": self._status,
            "last_error": self._last_error,
            "model": self.model_info,
            "blocked_sources": list(self.blocked_sources),
            "mitigation": [{"ip": ip, "stage": state.get("stage", "rate-limit")} for ip, state in self._mitigation.items()],
            "accuracy": {"TPR": tpr, "FPR": fpr, "TTD_sec": ttd},
            "kpi_counts": {"TP": self._tp, "FP": self._fp, "TN": self._tn, "FN": self._fn},
            "kpi_source": self._kpi_source,
            "prevented_downtime_min": self._prevented_downtime_min,
            "positive_rate": pos_rate,
        }

    def stop(self):
        self._stop_event.set()
        # Nudge external ingest loop to exit promptly
        try:
            if self._ingest is not None:
                self._ingest.put_nowait({"_stop": True})
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        # Note: captures are closed inside the worker thread to avoid asyncio loop errors on Windows
        self.cap = None
        self.cap_file = None
        self._status = "idle"

    def _resolve_model_path(self, path: str) -> str:
        """Resolve a model path. Accepts absolute paths, relative paths, or just a filename
        and tries common locations like the repo ./output folder.
        """
        candidates = []
        # 1) as provided (expanded)
        p = os.path.expanduser(path)
        candidates.append(p)
        # 2) if just a filename, try ./output/<name>
        if os.path.basename(p) == p:
            candidates.append(os.path.join(REPO_ROOT, "output", p))
        # 3) try repo root + provided (for relative paths from different CWDs)
        if not os.path.isabs(p):
            candidates.append(os.path.join(REPO_ROOT, p))
        # 4) try CWD ./output/<name>
        if os.path.basename(p) == p:
            candidates.append(os.path.join(os.getcwd(), "output", p))

        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(f"Model not found. Tried: {', '.join(candidates)}")

    def _resolve_source_path(self, source: str) -> str:
        """Resolve PCAP source path if it's a file. If it's an interface name, return as-is."""
        if source.endswith('.pcap'):
            s = os.path.expanduser(source)
            if os.path.isfile(s):
                return s
            # try repo root
            rr = os.path.join(REPO_ROOT, source)
            if os.path.isfile(rr):
                return rr
            # try sample-dataset in repo if only filename was given
            if os.path.basename(s) == s:
                sd = os.path.join(REPO_ROOT, 'sample-dataset', s)
                if os.path.isfile(sd):
                    return sd
            # try CWD
            cwdp = os.path.join(os.getcwd(), source)
            if os.path.isfile(cwdp):
                return cwdp
            # leave not found; pyshark will fail later with a clearer path
            return s
        return source

    def start(self, cfg: StartConfig):
        if self._status == "running":
            raise RuntimeError("Detector already running")

        # Determine source kind
        if cfg.source_kind in ("iface", "pcap", "http"):
            self._source_kind = cfg.source_kind
        else:
            self._source_kind = "pcap" if str(cfg.source).endswith('.pcap') else "iface"

        # Determine model path (explicit > env > auto-discover)
        model_path_candidate: Optional[str] = None
        if cfg.model_path and str(cfg.model_path).strip():
            model_path_candidate = str(cfg.model_path).strip()
        else:
            env_path = os.environ.get("LUCID_MODEL_PATH")
            if env_path and os.path.isfile(os.path.expanduser(env_path)):
                model_path_candidate = env_path
            else:
                auto = self._autodiscover_model_path()
                if auto:
                    model_path_candidate = auto
        if not model_path_candidate:
            raise FileNotFoundError("Model path not provided and no .h5 found. Set env LUCID_MODEL_PATH or place a model under ./output.")

        # Load model and infer time window and flow len from filename convention: '<t>t-<n>n-*.h5'
        resolved_model_path = self._resolve_model_path(model_path_candidate)
        model_filename = os.path.basename(resolved_model_path)
        try:
            prefix = model_filename.split('n')[0] + 'n-'
            self.time_window = int(prefix.split('t-')[0])
            self.max_flow_len = int(prefix.split('t-')[1].split('n-')[0])
        except Exception:
            # fallback to defaults if parsing fails
            self.time_window = 10
            self.max_flow_len = 10

        logger.info(f"Loading model: {resolved_model_path}")
        self.model = load_model(resolved_model_path)
        self.model_info = {
            "path": resolved_model_path,
            "time_window": self.time_window,
            "max_flow_len": self.max_flow_len,
        }

        # labels (optional - for accuracy calc if dataset info provided)
        # Try to auto-detect dataset type from source filename if not provided
        inferred_dataset = None
        try:
            src_name = os.path.basename(str(cfg.source)).lower()
            if any(tok in src_name for tok in ["cic-ddos-2019", "synflood", "udplag", "webddos", "dns"]):
                inferred_dataset = "DOS2019"
        except Exception:
            pass
        ds_type = cfg.dataset_type or inferred_dataset
        self.labels = parse_labels(ds_type, cfg.attack_net, cfg.victim_net)

        # capture setup or external ingest
        if self._source_kind == "http":
            self.cap = None
            self.cap_file = None
            # queue for external fragments
            self._ingest = queue.Queue()
            data_source = "http-ingest"
            logger.info(f"Starting external ingest mode (HTTP) (tw={self.time_window}s, n={self.max_flow_len})")
            self._kpi_source = "override" if (cfg.ground_truth_override in ("all_attack", "all_benign")) else "http-labels"
        else:
            resolved_source = self._resolve_source_path(cfg.source)
            logger.info(f"Starting capture: source={resolved_source} (tw={self.time_window}s, n={self.max_flow_len})")
            if resolved_source.endswith('.pcap'):
                # Defer opening FileCapture to the worker thread to avoid AnyIO "no current event loop" errors
                # in the request handling thread. We'll initialize the capture inside _loop with its own asyncio loop.
                self.cap = None
                self.cap_file = None
                data_source = os.path.basename(resolved_source)
                # PCAP playback settings from cfg or environment
                def _env_bool(name: str) -> Optional[bool]:
                    v = os.environ.get(name)
                    if v is None:
                        return None
                    return str(v).strip().lower() in ("1","true","yes","on")
                # Normalize PCAP path (absolute, forward slashes) to avoid Windows path quirks with tshark/pyshark
                try:
                    norm = os.path.abspath(resolved_source)
                except Exception:
                    norm = resolved_source
                if os.name == 'nt':
                    norm = norm.replace('\\', '/')
                self._pcap_path = norm
                self._pcap_realtime = bool(cfg.pcap_realtime) if (cfg.pcap_realtime is not None) else bool(_env_bool("LUCID_PCAP_REALTIME"))
                self._pcap_loop = bool(cfg.pcap_loop) if (cfg.pcap_loop is not None) else bool(_env_bool("LUCID_PCAP_LOOP"))
                logger.info("PCAP will be opened in detector thread with a dedicated event loop")
            else:
                # Preflight checks for live capture
                if not self._tshark_path():
                    raise ValueError("tshark not found. Install it (e.g., sudo apt install tshark) and ensure dumpcap has capture permissions (e.g., sudo dpkg-reconfigure wireshark-common; sudo usermod -aG wireshark $USER; or sudo setcap cap_net_raw,cap_net_admin=eip $(which dumpcap))")
                # Optional: hint if interface is unknown
                try:
                    names = {i.get('name') for i in self.list_interfaces()}
                    if names and resolved_source not in names:
                        logger.warning(f"Interface '{resolved_source}' not listed by tshark -D. Available: {sorted(names)}")
                except Exception:
                    pass
                try:
                    self.cap = pyshark.LiveCapture(interface=resolved_source)
                except Exception as e:
                    raise ValueError(
                        "Failed to open live capture on interface '%s': %s. "
                        "Verify interface exists (tshark -D), and that tshark/dumpcap can capture without root (see wireshark group or setcap)." % (resolved_source, e)
                    )
                self.cap_file = None
                data_source = resolved_source
            self._kpi_source = (
                "override" if (cfg.ground_truth_override in ("all_attack", "all_benign")) else ("dataset" if self.labels is not None else "none")
            )
        
        self.threshold = cfg.threshold
        self._gt_override = cfg.ground_truth_override
        # Reset KPI counters and mitigation state for a fresh run
        self._tp = self._fp = self._tn = self._fn = 0
        self._pos_total = 0
        self._samples_total = 0
        self._attack_active = False
        self._attack_start_ts = None
        self._first_alert_ts = None
        self._ttds = []
        self._window_index = 0
        self._attack_start_window = None
        self._history = []
        self.blocked_sources.clear()
        self._mitigation.clear()
        self._prevented_downtime_min = 0.0
        self._consecutive_alerts = 0
        self._ewma_ddos = None
        self._stop_event.clear()
        self._status = "running"
        self._last_error = None

        # start background loop
        self._thread = threading.Thread(target=self._loop, args=(data_source,), daemon=True)
        self._thread.start()

    def _loop(self, data_source: str):
        mins, maxs = static_min_max(self.time_window)
        # Create a dedicated asyncio event loop for this worker thread (required by pyshark in threads)
        worker_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            worker_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(worker_loop)
        except Exception:
            worker_loop = None
        try:
            while not self._stop_event.is_set():
                # Ingest traffic for this window
                if self._source_kind == "http":
                    # Drain ingest queue for the duration of time_window, with responsive stop
                    end = time.time() + self.time_window
                    fragments: List[Any] = []
                    src_ips: List[Optional[str]] = []
                    dest_ports: List[Optional[int]] = []
                    labels_list: List[Optional[int]] = []
                    if self._ingest is None:
                        # Sleep in small increments so stop is responsive
                        remaining = float(self.time_window)
                        while remaining > 0 and not self._stop_event.is_set():
                            step = 0.25 if remaining > 0.25 else remaining
                            time.sleep(step)
                            remaining -= step
                    else:
                        while time.time() < end and not self._stop_event.is_set():
                            try:
                                item = self._ingest.get(timeout=0.25)
                                if isinstance(item, dict) and item.get("_stop"):
                                    break
                                frags = item.get("fragments", []) or []
                                sips = item.get("src_ips") or [None] * len(frags)
                                dps = item.get("dest_ports") or [None] * len(frags)
                                lbs = item.get("labels") or [None] * len(frags)
                                fragments.extend(frags)
                                src_ips.extend(sips)
                                dest_ports.extend(dps)
                                labels_list.extend(lbs)
                            except queue.Empty:
                                pass
                    if self._stop_event.is_set():
                        break
                    if not fragments:
                        # nothing arrived this window; send heartbeat and continue
                        hb = {"ts": time.time(), "source": data_source, "samples": 0, "ddos_fraction": 0.0, "alert": False, "threshold": self.threshold}
                        if self._async_loop is not None:
                            try:
                                asyncio.run_coroutine_threadsafe(self.manager.broadcast(hb), self._async_loop)
                            except Exception:
                                pass
                        continue
                    # Build metrics for external fragments
                    uniq_dst_ports = {p for p in dest_ports if isinstance(p, int)}
                    uniq_src_ips = {ip for ip in src_ips if isinstance(ip, str) and ip}
                    packet_volume = int(sum(len(f) for f in fragments))
                    metrics = {
                        "flow_density": len(fragments),
                        "unique_dest_ports": len(uniq_dst_ports),
                        "src_ip_diversity": len(uniq_src_ips),
                        "packet_volume": packet_volume,
                    }
                    # Prepare model input X and keys
                    X_list = normalize_and_padding(fragments, mins, maxs, self.max_flow_len)
                    if len(X_list) == 0:
                        continue
                    X = np.array(X_list)
                    if X.size == 0 or X.ndim < 3:
                        continue
                    X = np.expand_dims(X, axis=3)
                    keys = [(src_ips[i] or "external", None, None, dest_ports[i] or 0, None) for i in range(len(fragments))]
                    Y_true = labels_list
                    external_packets = packet_volume
                else:
                    # Lazy-open PCAP capture in the worker thread with its own event loop
                    if self._source_kind == "pcap" and self.cap_file is None and self._pcap_path:
                        try:
                            # Respect explicit tshark path if provided
                            _kwargs = {"use_json": True, "keep_packets": False}
                            try:
                                _tp = os.environ.get("PYSHARK_TSHARK_PATH") or os.environ.get("TSHARK_PATH") or self._tshark_path()
                            except Exception:
                                _tp = None
                            if _tp:
                                _kwargs["tshark_path"] = _tp
                            self.cap_file = pyshark.FileCapture(self._pcap_path, **_kwargs)
                            logger.info(f"Opened PCAP in worker thread: {self._pcap_path}")
                        except Exception as e:
                            self._status = "error"
                            self._last_error = f"Failed to open PCAP with pyshark: {self._pcap_path} :: {e}"
                            break
                    cap = self.cap if self.cap is not None else self.cap_file
                    samples = process_live_traffic(cap, None, self.labels, self.max_flow_len, traffic_type="all", time_window=self.time_window)
                    if len(samples) == 0:
                        # For file capture, empty window may occur without EOF; only complete when EOF seen
                        if isinstance(cap, pyshark.FileCapture):
                            eof = False
                            err = None
                            try:
                                eof = bool(getattr(self.cap_file, "_lucid_eof", False))
                                err = getattr(self.cap_file, "_lucid_error", None)
                            except Exception:
                                eof = False
                                err = None
                            if err:
                                # Surface PCAP read errors (e.g., tshark missing) instead of silently completing
                                self._status = "error"
                                self._last_error = f"PCAP read error (tshark missing or crashed?): {err}"
                                break
                            if eof:
                                if self._pcap_loop and self._pcap_path:
                                    try:
                                        # Close and reopen to loop from beginning
                                        try:
                                            if self.cap_file:
                                                self.cap_file.close()
                                        except Exception:
                                            pass
                                        # Reopen in worker thread (honor tshark path if provided)
                                        _kwargs = {"use_json": True, "keep_packets": False}
                                        try:
                                            _tp = os.environ.get("PYSHARK_TSHARK_PATH") or os.environ.get("TSHARK_PATH") or self._tshark_path()
                                        except Exception:
                                            _tp = None
                                        if _tp:
                                            _kwargs["tshark_path"] = _tp
                                        self.cap_file = pyshark.FileCapture(self._pcap_path, **_kwargs)
                                        # reset indices/history if desired; keep history for charts
                                        self._window_index = 0
                                        logger.info(f"Looping PCAP from beginning: {self._pcap_path}")
                                    except Exception as e:
                                        self._status = "error"
                                        self._last_error = f"Failed to loop PCAP: {e}"
                                        break
                                else:
                                    self._status = "completed"
                                    break
                            # Not EOF; proceed to next window attempt
                            continue
                        # For live capture, just continue to next window
                        continue

                    # Apply mitigation simulation by filtering flows from blocked sources
                    if self.blocked_sources:
                        filtered = []
                        for (five_tuple, flow_dict) in samples:
                            src_ip = five_tuple[0]
                            if src_ip not in self.blocked_sources:
                                filtered.append((five_tuple, flow_dict))
                        samples = filtered

                    # Compute metrics for current window
                    metrics = self._compute_metrics(samples)

                    # Build input for model
                    X, Y_true, keys = dataset_to_list_of_fragments(samples)
                    if len(X) == 0:
                        # No fragments extracted this window; skip emitting
                        continue
                    X = np.array(normalize_and_padding(X, mins, maxs, self.max_flow_len))
                    if X.size == 0 or X.ndim < 3:
                        # Defensive: nothing to score
                        continue
                    X = np.expand_dims(X, axis=3)

                t0 = time.time()
                y_pred = np.squeeze(self.model.predict(X, batch_size=2048) > 0.5, axis=1)
                latency = time.time() - t0
                if self._source_kind == "http":
                    packets = external_packets
                else:
                    [packets] = count_packets_in_dataset([X])

                pos_count = int(np.sum(y_pred)) if y_pred.shape[0] > 0 else 0
                ddos_fraction = float(pos_count / y_pred.shape[0]) if y_pred.shape[0] > 0 else 0.0
                # Update cumulative prediction stats
                self._pos_total += pos_count
                self._samples_total += int(y_pred.shape[0])
                alert = ddos_fraction >= self.threshold

                # Ground-truth window label if available (any flow labeled ddos), or overridden
                ddos_true = None
                if self._gt_override in ("all_attack", "all_benign"):
                    ddos_true = (self._gt_override == "all_attack")
                elif len(Y_true) > 0 and any(v is not None for v in Y_true):
                    try:
                        arr = np.array([int(bool(v)) for v in Y_true if v is not None])
                        ddos_true = (np.sum(arr) > 0)
                    except Exception:
                        ddos_true = None

                # Update hysteresis counter
                if alert:
                    self._consecutive_alerts += 1
                else:
                    self._consecutive_alerts = 0

                auto_blocked_now: List[Dict[str, Any]] = []
                # Only act if hysteresis satisfied
                if alert and self._consecutive_alerts >= self.hysteresis_windows:
                    # Identify likely offending sources from predicted-positive fragments
                    counts: Dict[str, int] = {}
                    for idx, pred in enumerate(y_pred):
                        if pred:
                            src_ip = keys[idx][0]
                            if src_ip in self.allowlist:
                                continue
                            counts[src_ip] = counts.get(src_ip, 0) + 1
                    if counts:
                        # sort by descending count and act up to limit
                        top_sources = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                        to_consider = [src for src, _ in top_sources[: self._auto_block_limit]]
                        now = time.time()
                        for src in to_consider:
                            state = self._mitigation.get(src, {"stage": "monitor", "pos_count": 0, "last_action": 0.0, "cooldown_until": 0.0})
                            if now < state.get("cooldown_until", 0.0):
                                continue
                            state["pos_count"] = state.get("pos_count", 0) + counts[src]
                            # escalate stages based on positive evidence
                            if state["stage"] == "monitor":
                                state["stage"] = "rate-limit"
                            elif state["stage"] == "rate-limit" and state["pos_count"] >= 3:
                                state["stage"] = "blackhole"
                            state["last_action"] = now
                            state["cooldown_until"] = now + self.cooldown_seconds
                            self._mitigation[src] = state
                            # in this simulation, both stages are treated as blocked for training/demo
                            self.blocked_sources.add(src)
                            auto_blocked_now.append({"ip": src, "stage": state["stage"], "reason": counts[src]})

                # Downtime prevention estimation: if any mitigation is active during an alert window
                if alert and auto_blocked_now:
                    self._prevented_downtime_min += float(self.time_window) / 60.0

                # Predictive analytics via EWMA trend estimation
                if self._ewma_ddos is None:
                    self._ewma_ddos = ddos_fraction
                else:
                    self._ewma_ddos = self._ewma_alpha * ddos_fraction + (1 - self._ewma_alpha) * self._ewma_ddos
                predicted_next = self._ewma_ddos  # naive next-step forecast
                predicted_alert = predicted_next >= self.threshold

                # Confidence for ddos_fraction using Wilson interval
                def wilson(p, n, z=1.96):
                    if n == 0:
                        return (0.0, 0.0)
                    denom = 1 + z*z/n
                    center = p + z*z/(2*n)
                    margin = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5)
                    low = max(0.0, (center - margin)/denom)
                    high = min(1.0, (center + margin)/denom)
                    return (low, high)
                conf_low, conf_high = wilson(ddos_fraction, int(y_pred.shape[0]))

                # Explainability: feature z-scores vs rolling baseline
                baseline_count = min(30, len(self._history))
                feature_scores = {}
                if baseline_count > 5:
                    recent = self._history[-baseline_count:]
                    for k in ("flow_density", "unique_dest_ports", "src_ip_diversity", "packet_volume"):
                        vals = [h.get("metrics", {}).get(k, 0) for h in recent]
                        mean = float(np.mean(vals)) if vals else 0.0
                        std = float(np.std(vals)) if vals else 1.0
                        val = float(metrics.get(k, 0))
                        z = (val - mean) / (std if std > 0 else 1.0)
                        feature_scores[k] = z
                top_features = sorted(feature_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

                # Update accuracy statistics if ground truth available
                if ddos_true is not None:
                    if alert and ddos_true:
                        self._tp += 1
                    elif alert and not ddos_true:
                        self._fp += 1
                    elif (not alert) and ddos_true:
                        self._fn += 1
                    else:
                        self._tn += 1
                    # TTD (window-based): count windows between attack start and first alert
                    if ddos_true and not self._attack_active:
                        self._attack_active = True
                        self._attack_start_ts = time.time()
                        self._attack_start_window = self._window_index
                        self._first_alert_ts = None
                    if self._attack_active and alert and self._first_alert_ts is None and self._attack_start_window is not None:
                        self._first_alert_ts = time.time()
                        # At minimum, detection happens within the first attack window
                        windows_elapsed = max(1, self._window_index - self._attack_start_window + 1)
                        self._ttds.append(windows_elapsed * float(self.time_window))
                    if (not ddos_true) and self._attack_active:
                        # reset when attack window ends
                        self._attack_active = False
                        self._attack_start_ts = None
                        self._attack_start_window = None
                        self._first_alert_ts = None

                payload = {
                    "ts": time.time(),
                    "source": data_source,
                    "packets": int(packets),
                    "samples": int(y_pred.shape[0]),
                    "positive_count": pos_count,
                    "ddos_fraction": ddos_fraction,
                    "latency_sec": latency,
                    "metrics": metrics,
                    "alert": alert,
                    "threshold": self.threshold,
                    "blocked_sources": list(self.blocked_sources),
                    "mitigation": [{"ip": ip, "stage": st.get("stage", "rate-limit")} for ip, st in self._mitigation.items()],
                    "auto_blocked": auto_blocked_now,
                    "accuracy": {
                        "TPR": (self._tp / (self._tp + self._fn)) if (self._tp + self._fn) > 0 else None,
                        "FPR": (self._fp / (self._fp + self._tn)) if (self._fp + self._tn) > 0 else None,
                        "TTD_sec": (sum(self._ttds)/len(self._ttds)) if self._ttds else None,
                    },
                    "prevented_downtime_min": self._prevented_downtime_min,
                    "forecast": {
                        "ddos_fraction_next": predicted_next,
                        "predicted_alert": predicted_alert,
                        "confidence_low": conf_low,
                        "confidence_high": conf_high,
                    }
                }
                if top_features:
                    payload["explain"] = {"top_features": top_features}
                try:
                    logger.info(f"window ts={payload['ts']:.0f} src={data_source} samples={payload['samples']} pkts={payload['packets']} ddos_frac={ddos_fraction:.3f} alert={alert}")
                except Exception:
                    pass
                self._history.append(payload)
                # Limit history size
                if len(self._history) > 500:
                    self._history = self._history[-500:]

                # Broadcast to clients from a worker thread using the server's asyncio loop
                if self._async_loop is not None:
                    try:
                        asyncio.run_coroutine_threadsafe(self.manager.broadcast(payload), self._async_loop)
                    except Exception:
                        logger.exception("Failed to schedule broadcast to websocket clients")
                else:
                    # Loop not set yet; skip broadcast
                    logger.warning("Async loop not set; skipping broadcast for this window")

                # advance window index after completing this window
                self._window_index += 1

                # Optional: throttle PCAP playback to real-time windowing
                if self._source_kind == "pcap" and self._pcap_realtime and not self._stop_event.is_set():
                    # Sleep in short steps to be responsive to stop
                    remaining = float(self.time_window)
                    while remaining > 0 and not self._stop_event.is_set():
                        step = 0.25 if remaining > 0.25 else remaining
                        time.sleep(step)
                        remaining -= step
        except Exception as e:
            try:
                logger.exception("Error in detector loop")
            except Exception:
                pass
            self._last_error = str(e)
            self._status = "error"
        finally:
            # Ensure captures are closed in the same worker thread to avoid asyncio loop conflicts on Windows
            try:
                if self.cap_file is not None:
                    try:
                        self.cap_file.close()
                    except Exception:
                        pass
            finally:
                self.cap_file = None
            try:
                if self.cap is not None:
                    try:
                        self.cap.close()
                    except Exception:
                        pass
            finally:
                self.cap = None
            # Best-effort: close worker loop if we created one
            try:
                loop = asyncio.get_event_loop()
                if loop and loop is worker_loop:
                    try:
                        loop.stop()
                    except Exception:
                        pass
                    try:
                        loop.close()
                    except Exception:
                        pass
            except Exception:
                pass

    def _compute_metrics(self, samples: List):
        # samples: list of tuples (five_tuple, flow_dict)
        flow_count = len(samples)
        uniq_dst_ports = set()
        uniq_src_ips = set()
        total_pkts = 0
        for (five_tuple, flow_dict) in samples:
            src_ip, src_port, dst_ip, dst_port, proto = five_tuple
            if isinstance(dst_port, int):
                uniq_dst_ports.add(dst_port)
            uniq_src_ips.add(src_ip)
            # count packets in this window
            for k, arr in flow_dict.items():
                if k == 'label':
                    continue
                total_pkts += int(arr.shape[0])

        metrics = {
            "flow_density": flow_count,                # flows per window
            "unique_dest_ports": len(uniq_dst_ports),  # count of distinct destination ports
            "src_ip_diversity": len(uniq_src_ips),     # count of distinct source IPs
            "packet_volume": total_pkts,               # total packets in window
        }
        return metrics

    def block_sources(self, sources: List[str]):
        for s in sources:
            self.blocked_sources.add(s)

    def unblock_sources(self, sources: List[str]):
        for s in sources:
            if s in self.blocked_sources:
                self.blocked_sources.remove(s)

    def history(self) -> List[Dict[str, Any]]:
        return self._history


app = FastAPI(title="LUCID DDoS Detection Web Service", version="0.1.0")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("lucid-app")
manager = ConnectionManager()
service = DetectorService(manager)

def _load_demo_token() -> Tuple[Optional[str], str]:
    """Load demo auth token from environment or from a local file for convenience.
    Search order:
      1) DEMO_TOKEN environment variable
      2) Token file specified by DEMO_TOKEN_FILE env var (path)
      3) Repo-local files: .demo_token, demo_token.txt, config/demo_token.txt
    Returns the token string or None if not configured.
    """
    tok = os.environ.get("DEMO_TOKEN")
    if tok and tok.strip():
        logger.info("Demo auth enabled via DEMO_TOKEN environment variable")
        return tok.strip(), "env:DEMO_TOKEN"
    # Optional: externalize token in a file (avoids exporting every run)
    candidate_paths: List[str] = []
    env_file = os.environ.get("DEMO_TOKEN_FILE")
    if env_file:
        candidate_paths.append(os.path.expanduser(env_file))
    candidate_paths.extend([
        os.path.join(REPO_ROOT, ".demo_token"),
        os.path.join(REPO_ROOT, "demo_token.txt"),
        os.path.join(REPO_ROOT, "config", "demo_token.txt"),
    ])
    for p in candidate_paths:
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                    if val:
                        logger.info(f"Demo auth enabled via token file: {p}")
                        return val, f"file:{p}"
        except Exception as e:
            logger.warning(f"Failed reading DEMO token from {p}: {e}")
    return None, "none"

# Demo auth: require token for /api/* and /ws if configured
AUTH_TOKEN, AUTH_SOURCE = _load_demo_token()


# Serve static dashboard
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def on_startup():
    # Capture the running asyncio loop so background threads can dispatch coroutines
    loop = asyncio.get_running_loop()
    service.set_loop(loop)


@app.middleware("http")
async def demo_auth_middleware(request: Request, call_next):
    # Allow static and root without auth; protect /api/* when token is configured
    if AUTH_TOKEN and request.url.path.startswith("/api/"):
        token = None
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
        if not token:
            token = request.headers.get("x-auth-token")
        if not token:
            try:
                token = request.query_params.get("token")
            except Exception:
                token = None
        if not token:
            try:
                token = request.cookies.get("auth_token")
            except Exception:
                token = None
        if token != AUTH_TOKEN:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    # If demo auth enabled, require valid cookie/token before serving dashboard
    if AUTH_TOKEN:
        try:
            cookie_tok = request.cookies.get("auth_token")
            qp_tok = request.query_params.get("token") if request.query_params else None
            if (cookie_tok or qp_tok) != AUTH_TOKEN:
                return RedirectResponse(url="/login", status_code=303)
        except Exception:
            return RedirectResponse(url="/login", status_code=303)
    if os.path.exists(index_path):
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return FileResponse(index_path, headers=headers)
    return HTMLResponse("<h1>LUCID DDoS Dashboard</h1><p>UI not found.</p>", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.get("/login")
def login_page():
    path = os.path.join(TEMPLATES_DIR, "login.html")
    if os.path.exists(path):
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return FileResponse(path, headers=headers)
    return HTMLResponse("<h2>Login</h2><p>Missing template.</p>")


@app.post("/auth/login")
async def auth_login(request: Request, token: str = Form(...)):
    if not AUTH_TOKEN:
        # Auth disabled; accept any token and redirect
        resp = RedirectResponse(url="/", status_code=303)
        return resp
    if token != AUTH_TOKEN:
        return HTMLResponse("<h2>Unauthorized</h2><p>Invalid token.</p>", status_code=401)
    resp = RedirectResponse(url="/", status_code=303)
    # Set an HttpOnly cookie so API calls succeed without extra headers
    resp.set_cookie("auth_token", token, httponly=True, samesite="lax")
    return resp


@app.post("/auth/logout")
async def auth_logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("auth_token")
    return resp


@app.get("/api/status")
def get_status():
    return service.status()


@app.get("/api/auth/status")
def auth_status(request: Request):
    """Report whether demo auth is enabled and whether this request is authenticated.
    Useful for deployment verification and troubleshooting.
    """
    enabled = bool(AUTH_TOKEN)
    authed = False
    via = "none"
    if enabled:
        # Check cookie first (how the UI authenticates after login)
        try:
            cookie_tok = request.cookies.get("auth_token")
        except Exception:
            cookie_tok = None
        if cookie_tok and cookie_tok == AUTH_TOKEN:
            authed = True
            via = "cookie"
        else:
            # Also accept Authorization header for tooling
            auth = request.headers.get("authorization") or request.headers.get("Authorization")
            if auth and auth.lower().startswith("bearer "):
                token = auth.split(" ", 1)[1].strip()
                if token == AUTH_TOKEN:
                    authed = True
                    via = "header"
            if not authed:
                # Fallback: query param token
                try:
                    qp_tok = request.query_params.get("token")
                except Exception:
                    qp_tok = None
                if qp_tok and qp_tok == AUTH_TOKEN:
                    authed = True
                    via = "query"
    return {
        "enabled": enabled,
        "authenticated": authed,
        "auth_via": via,
        "source": globals().get("AUTH_SOURCE", "unknown"),
    }


@app.get("/api/history")
def get_history():
    return service.history()


@app.get("/api/interfaces")
def get_interfaces():
    """List available capture interfaces (best-effort)."""
    try:
        lst = service.list_interfaces()
        ok = True
    except Exception as e:
        lst = []
        ok = False
    return {"ok": ok, "interfaces": lst, "tshark": bool(service._tshark_path())}


def _pcap_search_roots() -> List[str]:
    roots = []
    # sample-dataset under repo
    sd = os.path.join(REPO_ROOT, "sample-dataset")
    if os.path.isdir(sd):
        roots.append(sd)
    # repo root
    roots.append(REPO_ROOT)
    # optional env var for extra dirs (comma or colon separated)
    extra = os.environ.get("LUCID_PCAP_DIRS")
    if extra:
        for part in extra.replace(";", ":").split(":"):
            p = part.strip()
            if p:
                roots.append(os.path.expanduser(p))
    # de-dup while preserving order
    seen = set()
    out = []
    for r in roots:
        rp = os.path.abspath(r)
        if rp not in seen and os.path.isdir(rp):
            seen.add(rp)
            out.append(rp)
    return out


@app.get("/api/pcaps")
def get_pcaps(limit: int = 200):
    """List available .pcap files from known folders (sample-dataset, repo root, optional env dirs)."""
    files: List[Dict[str, str]] = []
    try:
        roots = _pcap_search_roots()
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                # Skip hidden dirs to avoid scanning venv/.git, etc.
                base = os.path.basename(dirpath)
                if base.startswith('.') or base in ('.git', '.venv', 'venv', '__pycache__'):
                    continue
                for name in filenames:
                    if name.lower().endswith('.pcap'):
                        full = os.path.join(dirpath, name)
                        try:
                            rel = os.path.relpath(full, REPO_ROOT)
                        except Exception:
                            rel = full
                        files.append({"path": rel, "name": name})
                        if len(files) >= limit:
                            raise StopIteration
    except StopIteration:
        pass
    except Exception as e:
        return {"ok": False, "files": [], "error": str(e)}
    # sort by name for stable UX
    files.sort(key=lambda x: x.get("name", ""))
    return {"ok": True, "files": files}


@app.post("/api/start")
def start(cfg: StartConfig):
    try:
        service.start(cfg)
        return {"ok": True, "status": service.status()}
    except FileNotFoundError as e:
        # Bad request from client (wrong path)
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        # Problems opening PCAP/interface
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Conflicting state
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        # Unexpected server error
        logger.exception("Unhandled error in /api/start")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
def stop():
    service.stop()
    return {"ok": True, "status": service.status()}


class MitigationRequest(BaseModel):
    block_sources: Optional[List[str]] = None
    unblock_sources: Optional[List[str]] = None
    allowlist_add: Optional[List[str]] = None
    allowlist_remove: Optional[List[str]] = None
    hysteresis_windows: Optional[int] = None
    cooldown_seconds: Optional[int] = None


@app.post("/api/mitigation")
def mitigation(req: MitigationRequest):
    if req.block_sources:
        service.block_sources(req.block_sources)
    if req.unblock_sources:
        service.unblock_sources(req.unblock_sources)
    if req.allowlist_add:
        for ip in req.allowlist_add:
            service.allowlist.add(ip)
            # ensure allowlisted IPs are not blocked
            if ip in service.blocked_sources:
                service.blocked_sources.remove(ip)
            if ip in service._mitigation:
                del service._mitigation[ip]
    if req.allowlist_remove:
        for ip in req.allowlist_remove:
            if ip in service.allowlist:
                service.allowlist.remove(ip)
    if req.hysteresis_windows is not None:
        try:
            v = int(req.hysteresis_windows)
            if v >= 0:
                service.hysteresis_windows = v
        except Exception:
            pass
    if req.cooldown_seconds is not None:
        try:
            v = int(req.cooldown_seconds)
            if v >= 0:
                service.cooldown_seconds = v
        except Exception:
            pass
    return {
        "ok": True,
        "blocked_sources": list(service.blocked_sources),
        "allowlist": list(service.allowlist),
        "mitigation": [{"ip": ip, "stage": st.get("stage", "rate-limit")} for ip, st in service._mitigation.items()],
        "hysteresis_windows": service.hysteresis_windows,
        "cooldown_seconds": service.cooldown_seconds,
    }


@app.post("/api/ingest")
def ingest(data: IngestPayload):
    # Ensure service running and in external ingest mode
    if service._status != "running":
        raise HTTPException(status_code=409, detail="Detector not running")
    if service._source_kind != "http":
        raise HTTPException(status_code=409, detail="Ingest available only when started with source_kind='http'")
    if service._ingest is None:
        raise HTTPException(status_code=500, detail="Ingest buffer not initialized")
    # Validate metadata alignment
    n = len(data.fragments)
    if data.src_ips is not None and len(data.src_ips) != n:
        raise HTTPException(status_code=400, detail="src_ips length must match fragments length")
    if data.dest_ports is not None and len(data.dest_ports) != n:
        raise HTTPException(status_code=400, detail="dest_ports length must match fragments length")
    if data.labels is not None and len(data.labels) != n:
        raise HTTPException(status_code=400, detail="labels length must match fragments length")
    # Push into buffer; loop drains per time_window
    service._ingest.put({
        "fragments": data.fragments,
        "src_ips": data.src_ips,
        "dest_ports": data.dest_ports,
        "ts": data.ts or time.time(),
        "labels": data.labels,
    })
    return {"ok": True, "queued": n}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Enforce token via query param when configured
    if AUTH_TOKEN:
        try:
            qp = websocket.query_params or {}
            token = qp.get("token")
            if token != AUTH_TOKEN:
                try:
                    token = websocket.cookies.get("auth_token")
                except Exception:
                    token = None
            if token != AUTH_TOKEN:
                await websocket.close(code=1008)
                return
        except Exception:
            await websocket.close(code=1008)
            return
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection open; we don't expect messages from client, but we receive pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
