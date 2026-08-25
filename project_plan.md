# Project 1: E-commerce Data Analysis & Growth Dashboard

This project is a comprehensive Exploratory Data Analysis (EDA) and Business Intelligence application designed to highlight core data science and analytical skills on a large dataset (1M+ records).

## 🎯 Project Goals
1. **Clean messy data** (duplicates, missing fields, customer IDs, and outlier values).
2. **Engineer 8+ customer/product features** to enrich the dataset.
3. **Calculate 6 critical business KPIs** used by standard product/e-commerce teams.
4. **Build 10+ interactive and static visualizations** (Seaborn & Plotly).
5. **Develop an interactive Streamlit dashboard** featuring the KPIs, charts, and 5 core data-driven business insights (including statistical significance/experimentation analysis).

---

## 🛠️ Step-by-Step Implementation Strategy

### 🧹 Phase 1: Data Acquisition & Cleaning
* **Objective:** Prepare a high-quality dataset from raw, messy transaction logs.
* **Cleaning steps:**
  * Deal with null values (e.g., missing descriptions, unnamed customer IDs).
  * Handle transaction anomalies (negative quantities/prices representing refunds/cancellations).
  * Filter outliers using the IQR (Interquartile Range) method.

### ⚙️ Phase 2: Feature Engineering (8+ Features)
We will enrich the raw transactional data with the following calculated features:
1. **Recency:** Days since customer's last purchase.
2. **Frequency:** Total number of distinct orders placed by the customer.
3. **Monetary Value:** Total spend by the customer.
4. **Customer Lifetime Value (CLV):** Calculated customer value metric.
5. **Profit Margin:** Revenue minus cost (simulating a baseline product cost).
6. **Hour Bins:** Morning, Afternoon, Evening, Night classification of purchase time.
7. **Day Type:** Weekend vs. Weekday flag.
8. **Order Size Category:** Small, Medium, Bulk groupings.
9. **Customer Tenure:** Time duration between first and last purchase.

### 📈 Phase 3: KPI Metrics Calculation
We will write SQL and Python functions to compute:
1. **Total Revenue**
2. **Total Profit**
3. **Average Order Value (AOV)**
4. **Overall Profit Margin (%)**
5. **Monthly Active Customers (MAU)**
6. **Customer Retention Rate**

### 📊 Phase 4: Visualizations (10+ Charts)
* **Static (Seaborn/Matplotlib):**
  * Numerical feature correlation matrix.
  * Cohort retention rate heatmap.
  * RFM customer segment distribution.
* **Interactive (Plotly):**
  * Time-series revenue trend.
  * Sales & profit margin by country/region.
  * Purchase frequency vs. monetary value scatter plot.
  * Funnel drop-off visualization.
  * Hourly transaction bins distribution.
  * Category performance breakdown.
  * Top selling products Pareto chart.

### 🖥️ Phase 5: Streamlit Executive Dashboard
* **Structure:**
  * **Main Page:** KPI metrics cards (Revenue, Profit, AOV, etc.) and high-level charts.
  * **Customer Insights Tab:** RFM segmentations, cohort heatmap, and user behavior charts.
  * **Product Performance Tab:** Sales/margins by category and items.
  * **Experimentation Tab:** Interactive A/B testing calculator demonstrating statistical Z-tests/T-tests, p-values, and confidence intervals.
  * **Executive Report:** 5 key business insights written as a formal report.
