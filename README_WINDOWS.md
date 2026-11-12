# LUCID DDoS Detector — Windows Quick Start

This guide helps you run the LUCID web app on Windows 10/11 with live capture, PCAP playback, or external HTTP ingest.

## 1) Prerequisites

- Windows 10/11 (x64)
- Python 3.9–3.11 (recommended 3.10+)
- Git (optional, for cloning)
- Wireshark (includes tshark) + Npcap (for live capture)
- Microsoft Visual C++ Redistributable (usually installed automatically)

### 1.1 Install Python

Use the Microsoft Store or python.org. Ensure "Add Python to PATH" is checked.

```powershell
# Check version
py --version
python --version
```

### 1.2 Install Wireshark and Npcap

1) Download Wireshark for Windows from https://www.wireshark.org/
2) During setup:
   - Check "Install TShark".
   - Install Npcap when prompted.
   - If you want to capture without running as Administrator, UNCHECK "Restrict Npcap to Administrators only".
3) After install, add Wireshark to PATH if not already:
   - Control Panel → System → Advanced → Environment Variables → Path → add:
     - `C:\Program Files\Wireshark` (adjust if installed elsewhere)

Verify tshark works:

```powershell
tshark -v
# And list interfaces
tshark -D
```

If `tshark -D` shows interfaces (Ethernet/Wi-Fi/NPF_...), you’re good to go.

## 2) Get the code

```powershell
# Clone
git clone https://github.com/Nuwahereza-eng/lucid-ddos.git
cd lucid-ddos
```

Or download the ZIP from GitHub and extract.

## 3) Create a virtual environment (python39) and install deps

```powershell
# Create and activate a venv named "python39"
# Prefer Python 3.9 for widest TensorFlow 2.x compatibility on Windows
py -3.9 -m venv python39
.\python39\Scripts\activate

# Upgrade pip
python -m pip install -U pip

# Install runtime dependencies
pip install fastapi uvicorn[standard] pyshark numpy tensorflow==2.* pydantic
```

Notes:

- TensorFlow CPU is sufficient. GPU requires CUDA/cuDNN with matching versions (optional).
- If TensorFlow install is heavy, you can start with `pip install tensorflow-cpu==2.*`.

## 4) Run the web app

```powershell
# From the project root
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Open <http://127.0.0.1:8000> in your browser.
- Allow Python through Windows Defender Firewall when prompted.

## 5) Using the dashboard

### 5.1 Source types

- Network interface: Live capture using tshark/Npcap.
- PCAP file: Replay packets from a .pcap.
- External HTTP ingest: Push pre-parsed feature fragments to an HTTP endpoint.

### 5.2 Live capture (Network interface)

1) Click "Scan" next to Source to list interfaces (requires tshark).
2) Pick a normal interface (e.g., `Ethernet`, `Wi-Fi`, or an `NPF_{GUID}` entry).
3) Set Model path to your `.h5` model (the app expects names like `10t-10n-*.h5` so it can auto-detect window/flow length).
4) Optionally set Dataset type (e.g., `DOS2019`) so KPIs (TPR/FPR/TTD) are computed.
5) Click Start.

Troubleshooting:

- If Start fails, try running the terminal as Administrator OR reinstall Npcap with "Restrict Npcap to Admin only" UNCHECKED.
- Ensure `tshark -D` lists your interfaces.

### 5.3 PCAP playback

1) Set Source type to "PCAP file".
2) Set Source to a Windows path, e.g., `sample-dataset\CIC-DDoS-2019-SynFlood.pcap`.
3) Set Model path to your `.h5` file.
4) Set Dataset type to `DOS2019` (for CIC-DDoS-2019 PCAPs) to enable TPR/FPR/TTD.
5) Click Start.

### 5.4 External HTTP ingest (no packet capture)

This mode doesn’t sniff; you push JSON feature fragments to the app.

1) Set Source type to "External HTTP ingest" and click Start.
2) Use the dashboard’s Manual ingest panel to send demo fragments, or POST to the endpoint:

```powershell
# Using curl (Windows 10+ has curl)
curl -X POST http://127.0.0.1:8000/api/ingest ^
  -H "Content-Type: application/json" ^
  -d "{
    \"fragments\": [
      [[0.01,80,6,60,0,1],[0.02,80,6,52,0,1]],
      [[0.00,443,17,78,1,0]]
    ],
    \"src_ips\": [\"198.51.100.7\", \"203.0.113.10\"],
    \"dest_ports\": [80,443],
    \"labels\": [1,0]
  }"
```

If you include `labels` (0/1 per fragment), the app computes KPIs (TPR/FPR/TTD) in this mode.

## 6) KPIs and windows

- The model runs on fixed tumbling windows (default 10s), derived from the model filename (e.g., `10t-10n-…`).
- KPIs:
  - TPR: True Positive Rate over windows
  - FPR: False Positive Rate over windows
  - TTD: Time-To-Detect, measured in seconds as a multiple of the window size
- You’ll see the first chart point after one full window (~10s by default).
- You can adjust the alert threshold in the Start form; higher thresholds reduce FPR.

## 7) Common issues on Windows

- `Start failed: tshark not found`
  - Ensure Wireshark is installed and `C:\Program Files\Wireshark` is in PATH.
  - Restart your terminal after changing PATH.

- `Interface open failed` or `exit status` errors
  - Run PowerShell as Administrator OR reinstall Npcap with admin-only restriction unchecked to allow non-admin capture.
  - Verify `tshark -D` lists interfaces.

- TensorFlow install too big or slow
  - Use `pip install tensorflow-cpu==2.*` first. GPU can be added later if needed.

- No KPIs
  - Set Dataset type (e.g., `DOS2019`) for PCAP/iface or include `labels` in HTTP ingest.
  - You can also use the UI’s Ground truth override to force all windows Attack/Benign for sanity checks.

## 8) Development tips

- Virtual environment activation each session:

```powershell
.\python39\Scripts\activate
```

- Start server quickly:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- Stop server: Ctrl+C in the terminal. In HTTP mode, the dashboard Stop button is responsive within ~200ms.

## 9) Directory quick map

- `app/main.py` — FastAPI backend with REST + WebSocket
- `app/templates/index.html` — Dashboard UI
- `app/static/styles.css` — Styles and theming
- `lucid_cnn.py` — Model-related code (training/architecture)
- `lucid_dataset_parser.py` — Parsing/feature extraction utilities
- `util_functions.py` — Normalization, padding, metrics helpers
- `sample-dataset/` — Example PCAPs

## 10) Security & permissions

- Live capture requires Npcap. For least privilege, allow non-admin capture during Npcap install. Otherwise, run terminal as Administrator when capturing.
- The web app listens on 127.0.0.1 by default. Change to `--host 0.0.0.0` only when you intend to expose it and you’ve set up your firewall accordingly.

---
If you hit an error, copy the exact message (and whether you used interface/pcap/http) and we’ll resolve it quickly.
