Dual-Tower Evaluation Dashboard (`performance_app.py`)

This document explains provides clear instructions on how to successfully execute the Streamlit dashboard. 

## What was done in `performance_app.py`

The `app.py` script has been transformed from a prototype into a **production-grade evaluation dashboard** for our translation task assignment system.

## How to Execute the Dashboard Successfully

Follow these steps to run the Streamlit application on your local machine:

### Prerequisites
Install the necessary librarie: 
```bash
pip install streamlit pandas numpy torch plotly
```

### Verify Directory Structure
Ensure you have the following structure before running:
*   `performance_app.py` (The main application)
*   `demand.py` & `IDISC_DualTower.py` (Required modules)
*   `*.pth` (At least one trained model weights file must exist in the root folder alongside `performance_app.py`)
*   `Processed/` directory containing:
    *   `test_tasks.csv`
    *   `test_translators.csv`
    *   `test_labels.csv`
*   `Initial Dataset/CSV/` directory containing the raw CSVs (`Data.csv`, `Clients.csv`, `Schedules.csv`, `Translators Costs+Pairs.csv`, `Translators_Data.csv`).

### 2. Run the Application
Open the terminal , navigate to the folder  `performance_app.py`, and run the following command:

- streamlit run app.py

### 3. Using the App
1.  After running the command, Streamlit will automatically open a tab in your default web browser (usually at `http://localhost:8501`).
2.  **Sidebar**: Select the specific model weights (`.pth` file) you want to evaluate from the sidebar dropdown.
3.  **Per-Task Inspector**: Use the dropdown to select different tasks and see how the model and constraints perform on an individual level.
4.  **Global Analytics**: Switch to the second tab and click the **" Run Full Global Evaluation"** button to compute the Hit Rate and MRR over the 500 tasks and generate the charts. (takes a few minutes)
a