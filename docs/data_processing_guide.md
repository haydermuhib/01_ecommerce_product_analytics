# Data processing and feature engineering guide

This document outlines the data cleaning rules and feature engineering transformations applied to the raw online retail dataset during the ETL pipeline (`src/data_prep.py`).

---

## 1. Data cleaning steps

### Outlier capping
- **Method:** Calculated the 99.9th percentile threshold for both absolute quantity and unit price, filtering out records that exceeded these limits.
- **Why it matters:** E-commerce datasets contain transaction anomalies (such as bulk test orders of 10,000+ items or single unit prices over $5,000) that skew aggregate calculations and metrics. Capping preserves the core transactional baseline.

### Order cancellations (`IsCancelled`)
- **Method:** Evaluated whether the transaction `InvoiceNo` starts with the letter 'C' or if the `Quantity` is negative.
- **Why it matters:** Isolating cancellations prevents the dashboard from reporting inflated gross revenue and ensures return transactions do not bias baseline metrics.

### Guest checkouts (`CustomerID`)
- **Method:** Standardized missing customer identifiers by replacing them with the string `'Guest'`.
- **Why it matters:** Preserves valuable transactions (which would otherwise be dropped due to null constraints) while allowing cohort retention analysis to isolate registered profiles safely.

---

## 2. Feature engineering transformations

### Total sales (`TotalSales`)
- **Formula:** `Quantity * UnitPrice`
- **Why it matters:** Calculates the gross line-item revenue.

### Cost of goods sold (`COGS`)
- **Formula:** `Quantity * (UnitPrice * 0.60)`
- **Why it matters:** Simulates a standard 60% operational acquisition cost since the raw dataset only provides public selling prices.

### Net profit (`Profit`)
- **Formula:** `TotalSales - COGS` (representing a constant 40% margin on active sales).
- **Why it matters:** Tracks the direct profit margin per transaction.

### Hourly purchase bins (`HourBin`)
- **Method:** Grouped transaction times into four distinct shifts:
  - Night: 0:00 to 6:00
  - Morning: 6:00 to 12:00
  - Afternoon: 12:00 to 18:00
  - Evening: 18:00 to 24:00
- **Why it matters:** Helps marketing analysts identify peak buying hours for scheduling promotional campaigns.

### Day type (`DayType`)
- **Method:** Classified transactions as either `'Weekend'` (Saturday/Sunday) or `'Weekday'` (Monday-Friday).
- **Why it matters:** Highlights weekly cyclical consumer buying habits.

### Order size category (`OrderSize`)
- **Method:** Partitioned transaction quantities into size categories:
  - Small: 1 to 5 units
  - Medium: 6 to 24 units
  - Bulk: 25+ units
- **Why it matters:** Segments consumer behavior sizes (e.g., individual buyers vs. small businesses).

### Hemisphere classification (`Hemisphere`)
- **Method:** Checked if the transaction country belongs to:
  - `["Australia", "New Zealand", "South Africa", "Brazil"]`
  - If yes, mapped to `'Southern'`. Otherwise, mapped to `'Northern'`.
- **Why it matters:** Resolves seasonal demand patterns accurately. Southern Hemisphere transactions use inverted seasonal mappings (e.g., Summer in December/January/February) so that product sales trends reflect physical seasons correctly.
