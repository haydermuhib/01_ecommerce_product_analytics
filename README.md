# E-Commerce Product Analytics and Experimentation Dashboard
## Technical portfolio presentation

**Live Application:** https://01ecommerce-analytics.streamlit.app/
**Duration:** 10 minutes review
**Audience:** Technical Recruiters and Hiring Managers
**Date:** 2026-10-03

---

## Agenda

1. Business problem and executive summary (2 min)
2. Data dictionary and schema harmonization (2 min)
3. Tech stack (1 min)
4. System architecture and pipeline flow (2 min)
5. Core analytical findings and takeaways (2 min)
6. Installation and reproduction commands (1 min)

---

## 1. Business problem and executive summary

E-commerce businesses frequently struggle with high customer acquisition costs paired with rapid post-purchase churn. Without visibility into customer tenure, purchasing cadence, and product-level demand, commercial teams waste budget on broad marketing rather than targeted retention.

This project delivers an end-to-end analytical solution processing over 540,000 real-world retail transactions. It models customer lifetime value, evaluates checkout redesigns using statistical hypothesis testing, and flags churn risk with predictive machine learning.

### Analytical Scope and Business Discipline
This project represents a dedicated **Product Growth Analytics and Predictive Experimentation** suite designed for product-led growth teams and data scientists:

- **Analytical Discipline:** Supervised Machine Learning (Churn Classification), Statistical Hypothesis Testing (Two-Tailed Z-Tests), and Product Growth Simulation.
- **Core Business Questions Answered:**
  1. Which customers are at imminent risk of churning, and what behavioral features most strongly drive retention failure?
  2. Did a checkout funnel intervention produce a statistically significant conversion lift, and did the experiment achieve adequate statistical power?
  3. What are the minimum detectable effect sizes and required sample sizes before launching new product features?
- **Target Stakeholders:** Product Managers, Growth Data Scientists, Experimentation Leads, and Conversion Rate Optimization (CRO) Engineers.

### Key project outcomes
- Cleaned and harmonized 541,909 raw transaction logs, isolating returns, credit cancellations, and guest sessions into an optimized SQLite analytical database.
- Implemented monthly customer cohort tracking, uncovering a 60% retention drop within the first 30 days that stabilizes into predictable long-term buying behavior.
- Built an RFM segmentation engine revealing that VIP Champions generate over 35% of total revenue despite making up less than 8% of the customer base.
- Developed an interactive A/B testing simulator computing two-sample Z-proportion hypothesis tests, critical rejection boundaries, and statistical power curves.
- Created a Random Forest churn risk classifier deployed in a tactile Neumorphic Dark Streamlit dashboard with visualizations built exclusively using Matplotlib and Seaborn Object-Oriented APIs.

---

## 2. Data dictionary and schema harmonization

The raw data source is the UCI Online Retail transaction dataset. Transactional edge cases including cancellations ('C' prefix), negative quantities, zero unit prices, and missing customer identifiers were cleaned and stored into SQLite tables:

<details>
<summary><b>Click to expand database schema and field definitions</b></summary>

### Transactions table schema (`ecommerce.db`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | TEXT | Unique transaction identifier (values starting with 'C' indicate cancellations) |
| `StockCode` | TEXT | Unique inventory product identifier |
| `Description` | TEXT | Normalized item title with whitespace trimmed |
| `Quantity` | INTEGER | Item units purchased (capped at 99.9th percentile) |
| `InvoiceDate` | TIMESTAMP | Full transaction timestamp |
| `UnitPrice` | REAL | Price per unit in USD (capped at 99.9th percentile) |
| `CustomerID` | TEXT | Customer identifier (missing values mapped to 'Guest') |
| `Country` | TEXT | Customer geographic country of origin |
| `IsCancelled` | BOOLEAN | Boolean flag denoting cancellation or return transactions |
| `TotalSales` | REAL | Computed transaction gross revenue (`Quantity * UnitPrice`) |
| `COGS` | REAL | Simulated cost of goods sold (60% baseline) |
| `Profit` | REAL | Net transaction profit (`TotalSales - COGS`) |
| `HourBin` | TEXT | Categorized transaction timing (Morning, Afternoon, Evening, Night) |
| `DayType` | TEXT | Categorized day of purchase (Weekday vs Weekend) |
| `Hemisphere` | TEXT | Geographic classification for season alignment |

### Customers table schema (`ecommerce.db`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | TEXT | Unique customer identifier |
| `Recency` | INTEGER | Elapsed days since most recent purchase |
| `Tenure` | INTEGER | Total active days between first and last recorded order |
| `Frequency` | INTEGER | Count of distinct completed invoice orders |
| `Monetary` | REAL | Total cumulative lifetime spend |
| `CohortMonth` | TEXT | Year and month of customer's first purchase |
| `Segment` | TEXT | Assigned behavioral tier (VIP Champions, Loyal Customers, Recent New Buyers, At Risk, Lost) |

</details>

---

## 3. Tech stack

- **Language:** Python (tested on Python 3.10 and newer)
- **Package and Environment Manager:** `pixi` (`pixi.toml` and lockfile)
- **Data Engineering and Processing:** `pandas`, `numpy`, `sqlite3`, `sqlalchemy`
- **Visualization:** `matplotlib` (Object-Oriented API), `seaborn`
- **Statistical Testing:** `scipy.stats` (two-sample Z-proportion tests, power calculations)
- **Machine Learning:** `scikit-learn` (Random Forest Classifier, train-test splitting, metrics)
- **Dashboard Interface:** `streamlit`
- **Exploratory Walkthrough:** `notebooks/01_ecommerce_eda.ipynb`
- **Visual Design Standard:** Tactile Neumorphic Dark Theme (`Design.md`)

---

## 4. System architecture and pipeline flow

The system enforces strict layer separation. Analytics and UI layers treat processed SQLite databases as read-only.

```text
                           [Raw Transaction Logs]
                        (data/raw/online_retail.csv)
                                     │
                                     ▼ (pixi run data-prep)
                           [data/processed/ecommerce.db]
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
[notebooks/01_ecommerce_eda.ipynb] [src/train_model.py]       [streamlit_app.py Dashboard]
(Human exploratory analysis)        (Random forest model)      (Streamlit presentation UI)
                                             │                           ▲
                                             ▼                           │
                                  [models/churn_model.pkl] ──────────────┘
```

<details>
<summary><b>Click to expand modular code details</b></summary>

- `data/download_data.py`: Downloads raw online retail records from the data repository.
- `notebooks/01_ecommerce_eda.ipynb`: Documents the complete exploratory data analysis narrative, schema checks, distribution visualizations, and initial cohort matrices.
- `src/data_prep.py`: Cleans raw records, handles guest users, caps 99.9th percentile outliers, engineers financial features, and creates optimized SQLite tables.
- `src/data_processing.py`: Provides pure mathematical helpers for cohort retention calculations and seasonal product aggregations.
- `src/stats_helpers.py`: Implements two-sample Z-proportion testing, statistical power estimations, and normal distribution curve coordinates.
- `src/train_model.py`: Trains and serializes the customer churn prediction pipeline.
- `streamlit_app.py`: Presents an interactive, multi-tab Neumorphic Dark dashboard with real-time sidebar filters and Matplotlib/Seaborn Object-Oriented visual rendering.

</details>

---

## 5. Core analytical findings and takeaways

### Customer retention and the month-one cliff
- Across all acquisition cohorts, customer retention drops by over 60% within the first 30 days after signup.
- By month three, cohort retention stabilizes between 15% and 20%, representing the core loyal customer segment.
- Early retention campaigns and onboarding incentives deployed at day 25 have the highest probability of preventing permanent customer churn.

### Revenue concentration and RFM distribution
- The VIP Champions customer segment represents less than 8% of all registered buyers but produces more than 35% of total cumulative store sales.
- Customers in the At Risk segment have high historical frequency but have remained inactive for over 150 days on average, indicating prime candidates for automated re-engagement.

### Product catalog dynamics and geographic expansion
- A Pareto distribution analysis confirms that the top 20% of catalog items generate roughly 80% of total revenue.
- While the United Kingdom accounts for the largest aggregate order count, European destinations including Germany and France exhibit 12% higher Average Order Value (AOV).

---

## 6. Installation and reproduction commands

You need Python 3.10 or newer and `pixi` installed.

### Setup commands
1. Navigate to the project directory:
   ```bash
   cd 01_ecommerce_product_analytics
   ```
2. Install project dependencies using pixi:
   ```bash
   pixi install
   ```
3. Ingest raw retail data, run ETL pipeline, and train the ML classifier:
   ```bash
   pixi run download-data
   pixi run data-prep
   pixi run train-model
   ```
4. Launch the local interactive dashboard:
   ```bash
   pixi run app
   ```

---

## Quick reference card

### Standard execution commands
| Operation | Command |
| :--- | :--- |
| Environment Setup | `pixi install` |
| Download Dataset | `pixi run download-data` |
| Run Data Prep Pipeline | `pixi run data-prep` |
| Train ML Churn Model | `pixi run train-model` |
| Start Presentation App | `pixi run app` |
| Exploratory Notebook | `notebooks/01_ecommerce_eda.ipynb` |

### Key output artifacts
- SQLite Database: `data/processed/ecommerce.db`
- Model Binary: `models/churn_model.pkl`
- Model Performance Metrics: `models/metrics.txt`
- Exploratory Notebook: `notebooks/01_ecommerce_eda.ipynb`
- Presentation Dashboard: `streamlit_app.py`
