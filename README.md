🇧🇭 Bahrain Statistical AI Agent
Automated Data Ingestion · Master Dataset Builder · AI Query Engine · Extensible Analytics Platform
📌 Overview

The Bahrain Statistical AI Agent is an end-to-end system that automatically:

Fetches datasets from government open data portals

Cleans & normalizes them

Merges them safely into unified master datasets

Answers questions using both rule-based logic and LLM (ChatGPT) fallback

Allows clients to drag-and-drop CSV files into data/incoming/

Supports fully automated 6-month scheduled updates

Easily extends into segmentation, forecasting, or mobility analytics

This repository is designed so that even non-technical users can maintain updated national statistics without breaking anything.

📂 Project Structure
   bahrain_stats_agent/
│
├── bahrain_agent/
│   ├── agent.py               # Core reasoning engine
│   ├── nlu_router.py          # Intent detection + LLM fallback
│   ├── describe_layer.py      # Statistical descriptions & summaries
│   ├── repo.py                # Repository for master datasets
│
├── scripts/
│   ├── fetch_and_ingest.py    # Automatic dataset downloader + ingestion trigger
│   ├── ingest_and_prepare.py  # Cleansing, normalization, merging pipeline
│   ├── webhook_receiver.py    # Optional real-time ingestion API
│
├── config/
│   ├── endpoints.json         # URL list for fetching datasets
│   ├── schemas.json           # Schema mapping rules for ingestion
│
├── data/
│   ├── incoming/              # Drop new CSVs here (manual or automatic)
│   ├── incoming_failed/       # Suspicious files stored safely
│   ├── master/                # Final cleaned, unified datasets
│
├── logs/                      # Automatic logs from fetching & ingestion
│
├── architecture.md            # System design & architecture
└── README.md                  # This file

🧠 Key Features
✔ Automated Data Ingestion

Downloads datasets from URLs in endpoints.json

Supports retries, file-size limits, deduplication

Performs CSV validation and rejects suspicious files

Automatically triggers ingestion pipeline

✔ Intelligent Data Normalization

The ingestion pipeline handles:

Column name unpredictability

Nationality / governorate normalization

Year extraction & parsing

Duplicates and row-level validation

Safe merging into master datasets

✔ Drag-and-Drop Data Support

Client can simply drop CSVs into:
data/incoming/
Then run:
python scripts/ingest_and_prepare.py --run

No formatting needed — the system will adapt.

✔ Hybrid AI Agent

Combines:

Rule-based logic

Structured dataset querying

Large Language Model fallback

Query refinement

Year detection

Intent classification

✔ Future-Ready Architecture

Built to support:

Housemaid demand segmentation

Labour diagnosis modeling

Mobility pattern integration

Workforce surge/shortage predictions

Nationality cluster analysis

⚙️ Installation
1. Clone the repository
   git clone https://github.com/yourname/bahrain_stats_agent.git
cd bahrain_stats_agent
2. Create a virtual environment
   python -m venv venv
3. Activate the environment
   venv\Scripts\activate
4. Install dependencies
   pip install -r requirements.txt

🚀 Usage Guide
1. Automatic Fetch + Ingestion

Fetch datasets from all URLs in config/endpoints.json:

python scripts/fetch_and_ingest.py --run


Optional dry run (no data is written to master files):

python scripts/fetch_and_ingest.py --run --dry

2. Manual Drag-and-Drop Workflow

Place CSVs into:

data/incoming/


Run ingestion:

python scripts/ingest_and_prepare.py --run


This safely updates master datasets.

3. Automated Every 6 Months (Windows Task Scheduler)
Create a scheduled task:

Program: python

Arguments:

scripts/fetch_and_ingest.py --run


Start in:

C:\path\to\bahrain_stats_agent\


Trigger:
Every 6 months

Your client never needs to do anything manually again.

📡 Optional: Webhook Ingestion

You can run an ingestion API server:

uvicorn scripts.webhook_receiver:app --host 0.0.0.0 --port 8000


Supports:

Upload CSV (multipart)

Provide URL for auto-download

Send raw CSV text

Trigger ingestion via HTTP

📝 Config Files
endpoints.json

Stores URLs for fetching:

{
  "endpoints": [
    "https://data.gov.bh/…/population.csv",
    "https://data.gov.bh/…/labour.csv"
  ]
}

schemas.json

Stores flexible mappings:

{
  "synonyms": {
    "nationality": ["nat", "nation", "n"],
    "governorate": ["gov", "region", "muharraq"]
  }
}


The ingestion pipeline uses this file to detect how to map incoming CSVs.

📊 Master Dataset Philosophy

Each master CSV is designed to be:

Non-destructive

Append-safe

Schema-validated

Human-readable

Machine-consumable

This ensures long-term consistency even with unstable data sources.

🧩 Extending the Model

You can easily plug new modules into:

describe_layer.py
agent.py
repo.py

Examples:

➤ Housemaid Demand Segmentation

Add describe_domestic_workers() + domain logic.

➤ Labour Market Forecasting

Add forecast_labour() using statsmodels or ML.

➤ Mobility Integration

Add mobility_segmentation() using telecom movement files.

📈 Example Query Flow

User asks:
“Give me population density in Muharraq for 2020.”

nlu_router.py:

Detects entity → population density

Extracts year → 2020

Passes to agent.py

agent.py:

Loads density master dataset

Filters for Muharraq, 2020

Formats structured answer

If missing values → falls back to LLM with safe context.

🧪 Development Notes

To work safely:

Dry Run:
  
  python scripts/ingest_and_prepare.py --run --dry

Verbose mode:

Enable detailed logging in /logs/.

Duplicate Safety:

The system:

Detects duplicates using MD5 hashing

Keeps the older file

Skips unwanted duplicates

Ensures master dataset stability





