# E-Commerce Product Analytics Dashboard
## Technical portfolio presentation

**Live Dashboard Application:** [01ecommerce-analytics.streamlit.app](https://01ecommerce-analytics.streamlit.app/)

**Duration:** 10 minutes review
**Audience:** Technical Recruiters and Hiring Managers
**Date:** 2026-08-29

---

## Agenda

1. Project overview and features (2 min)
2. Database schema and dataset details (2 min)
3. Tech stack (1 min)
4. System architecture and codebase layout (3 min)
5. Installation and verification (2 min)

---

## 1. Project overview and features

This application processes online retail transaction logs, groups customers into behavioral segments, evaluates A/B test results, and predicts customer churn.

### Core modules
- The Executive Overview tracks transaction volumes, sales revenue, average order value, profit margins, and hourly buying peaks.
- Customer Segmentation displays dynamic cohort retention percentages and RFM customer value treemaps.
- Product Performance identifies top-selling inventory items using Pareto charts, and maps seasonal demand using hemisphere-aware date partitioning.
- The A/B Testing Simulator calculates z-proportion conversions, statistical power curves, and critical decision boundaries.
- The Predictive Churn Form runs real-time Random Forest predictions to assess customer churn risk.

---

## 2. Database schema and dataset details

The application processes the UCI Online Retail transaction dataset containing 541,909 raw records.

<details>
<summary><b>Click to expand database schema and variables</b></summary>

### Transactions table schema
- `InvoiceNo` represents the unique transaction identifier (values starting with 'C' indicate cancellations).
- `StockCode` represents the unique product item code.
- `Description` represents the item description.
- `Quantity` represents the number of units purchased per transaction.
- `InvoiceDate` represents the sale timestamp.
- `UnitPrice` represents the product price per unit.
- `CustomerID` represents the customer identifier (labeled as 'Guest' for guest checkouts).
- `Country` represents the customer's country of residence.
- `TotalSales` represents the engineered transaction revenue (`Quantity * UnitPrice`).
- `COGS` represents the simulated Cost of Goods Sold (60% of price).
- `Profit` represents the simulated profit margin (`TotalSales - COGS`).
- `Hemisphere` represents the engineered hemisphere tag (Northern or Southern) used for season mapping.

</details>

---

## 3. Tech stack

- **Tools:** SQLite3, VS Code
- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy`, `scikit-learn`, `plotly`, `streamlit`, `ipykernel`

---

## 4. System architecture and codebase layout

The application maps raw data through structured ETL processing, exports a trained predictive model, and serves Plotly visualizations:

```text
[Raw Dataset]
      │
      ▼ (data/download_data.py)
[data/raw/online_retail.csv]
      │
      ▼ (src/data_prep.py ETL & Feature Engineering)
[data/processed/ecommerce.db] ──┐
      │                         │
      │ (loads features)        │ (queries tables)
      ▼                         ▼
(src/train_model.py)         (app.py Streamlit UI) ◄── (src/data_processing.py & stats_helpers.py)
      │                         ▲
      ▼ (exports pipeline)      │ (loads classifier)
[models/churn_model.pkl] ───────┘
```

<details>
<summary><b>Click to expand modular code details</b></summary>

- `data/download_data.py` pulls the raw CSV transactions from the remote repository.
- `src/data_prep.py` runs the ETL cleaning script, caps outlier quantities, classifies countries by hemisphere, and creates the SQLite tables.
- `src/train_model.py` trains the random forest classifier on RFM metrics to predict customer churn risk (defined as 90 days of inactivity) and serializes the model.
- `models/metrics.txt` holds the logged classification report and ROC-AUC metrics from the latest training run.
- `src/data_processing.py` calculates cohort retention matrices and groups seasonal product metrics.
- `src/stats_helpers.py` computes statistical power, z-scores, critical regions, and p-values for A/B testing evaluation.
- `app.py` renders the Neumorphic Dark Theme Streamlit layout and passes variables to Plotly charts.

</details>

---

## 5. Installation and verification

You need Python 3.10 or newer and the `uv` package manager installed.

### Setup commands
1. Navigate to the project root:
   ```bash
   cd 01_ecommerce_product_analytics
   ```
2. Synchronize the virtual environment and dependencies:
   ```bash
   uv sync
   ```
3. Ingest raw retail data, run ETL pipeline, and train the ML classifier:
   ```bash
   uv run python data/download_data.py
   uv run python src/data_prep.py
   uv run python src/train_model.py
   ```
4. Launch the local Streamlit dashboard server:
   ```bash
   uv run streamlit run app.py
   ```

<details>
<summary><b>Click to expand local troubleshooting steps</b></summary>

**Port 8502 already occupied:**
```bash
fuser -k 8502/tcp
```

**Virtual environment python path resolution warnings in VS Code:**
Verify that `.vscode/settings.json` points to the absolute path of the `.venv/bin/python` interpreter.

</details>

---

## Quick reference card

### Standard execution commands
| Operation | Command |
|-----------|---------|
| Download Dataset | `uv run python data/download_data.py` |
| Run Data Prep | `uv run python src/data_prep.py` |
| Train ML Model | `uv run python src/train_model.py` |
| Start App Server | `uv run streamlit run app.py` |

### Important file paths
- SQLite Database: `data/processed/ecommerce.db`
- Model Binary: `models/churn_model.pkl`
- Model metrics: `models/metrics.txt`
- Main Dashboard App: `app.py`

