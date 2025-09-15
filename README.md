# Diabetes Detection — Streamlit App

This app loads a trained RandomForest pipeline (from Colab) to score diabetes risk
using NHANES clinical features + wearable sensor aggregates.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
