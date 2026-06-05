# iDISC AssignMate HUD — Frontend Setup Guide

This is the frontend dashboard for the iDISC AssignMate project. It shows the full Translators and Clients database and lets PMs upload a CSV of pending tasks to get AI-powered assignment recommendations.

---

## ⚡ Quick Start (2 commands)

> **Requires Python 3** (any version). No extra libraries needed.

```bash
# 1. Navigate to the frontend folder
cd frontend

# 2. Start the local server (opens the browser automatically)
python serve.py
```

The app will open at **http://localhost:8080/idisc_hud.html**.

---

## ❓ Why can't I just open the HTML file directly?

If you double-click `idisc_hud.html` to open it in your browser, it will load the UI but the **Translators and Clients databases will be empty**. This is because browsers block local file-loading (`fetch()` over `file://`) for security reasons.

You must serve the app through a local HTTP server — that's exactly what `python serve.py` does.

---

## 📁 Folder Structure

```
frontend/
├── idisc_hud.html        ← The main app (open this via the server)
├── serve.py              ← Local server launcher (run this!)
├── extract_data.py       ← Regenerates the JSON files from the CSV dataset
├── extract_data.js       ← Same as above but Node.js version
└── data/
    ├── translators.json  ← ~900 translators pre-extracted from the dataset
    └── clients.json      ← ~2600 clients pre-extracted from the dataset
```

---

## 🔄 How to Regenerate the Database Files

The `data/translators.json` and `data/clients.json` files are already included in the repo (pre-built). You only need to run `extract_data.py` if:
- The source dataset has changed, or
- The JSON files are missing from your clone.

**Before running**, you need the dataset CSV. Follow the steps in `DATA/README.md` to generate `DATA/Initial Dataset/CSV/Data.csv` from the source zip file.

Once you have `Data.csv`, run:

```bash
# From the repo root
python frontend/extract_data.py
```

This reads `DATA/Initial Dataset/CSV/Data.csv` and regenerates:
- `frontend/data/translators.json`
- `frontend/data/clients.json`

---

## 🧭 Using the App

| Section | What it does |
|---|---|
| **Tasks** (default) | Upload a CSV of pending tasks via the ↑ button; get AI translator recommendations |
| **Translators** | Browse and search the full translator roster with stats and language pairs |
| **Clients** | Browse and search all clients with industry classifications and SLA info |
| **Dashboard** | Summary metrics (static placeholder for now) |

### Uploading Tasks

Click the **↑ upload button** in the Pending Tasks panel. The CSV must have headers that match the dataset columns. Supported column names:

| CSV Header | Maps to |
|---|---|
| `task_id` or `id` | Task ID |
| `task_type` | Translation / ProofReading / PostEditing |
| `source_lang` | Source language |
| `target_lang` | Target language |
| `manufacturer` | Client name |
| `manufacturer_industry` | Industry classification |
| `hours` | Forecasted hours |
| `end` | Deadline |

---

## 🔌 Connecting to the Real Backend (Automatic)

The frontend HUD features **automatic server auto-detection**:
1. Simply start the FastAPI inference backend (e.g. `python backend/inference_server.py` on port `8000`).
2. When you refresh or load the dashboard in the browser, the frontend script automatically pings the server.
3. If detected, it immediately establishes a connection and transitions to **Online Mode** (switching the server status indicator badge in the header from grey "Offline Mode" to green "[MODEL_ID] Active"), pulling real-time recommendations, model configurations, and saving registered translators back to disk.
4. If the server is offline or unreachable, the dashboard automatically falls back to **Offline Mode**, computing matching scores locally in the browser using the baseline client-side engine.

---

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| App opens but Translators/Clients are empty | You opened `idisc_hud.html` directly. Run `python serve.py` instead. |
| Port 8080 already in use | Open `serve.py` and change `PORT = 8080` to any free port, e.g. `8081`. |
| `python` command not found | Try `python3 serve.py`. On Windows, check that Python is in your PATH. |
| JSON files are missing | Run `python frontend/extract_data.py` (requires the Data.csv — see `DATA/README.md`). |
