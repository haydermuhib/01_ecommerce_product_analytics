import os
import pandas as pd
import numpy as np
import sqlite3

# Define paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RAW_CSV_PATH = os.path.join(PROJECT_DIR, "data", "raw", "online_retail.csv")
PROCESSED_DB_PATH = os.path.join(PROJECT_DIR, "data", "processed", "ecommerce.db")

def run_data_pipeline():
    print("Starting Data Cleaning and Feature Engineering Pipeline...")
    
    # Ensure raw data exists
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_CSV_PATH}. Please run download_data.py first.")
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(PROCESSED_DB_PATH), exist_ok=True)
    
    # 1. Load data
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"Loaded {df.shape[0]:,} rows.")
    
    # ------------------ PHASE 1: DATA CLEANING ------------------
    print("\nPhase 1: Data Cleaning...")
    
    # Remove duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {dup_count:,} duplicate rows.")
        
    # Deal with missing descriptions
    df['Description'] = df['Description'].fillna("Unlabeled Item").str.strip()
    
    # Cast InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed')
    
    # Identify Cancellations (InvoiceNo starting with 'C' or negative Quantity)
    df['IsCancelled'] = df['InvoiceNo'].astype(str).str.startswith('C') | (df['Quantity'] < 0)
    
    # Handle missing Customer IDs: Label them as 'Guest'
    df['CustomerID'] = df['CustomerID'].fillna(-1).astype(int).astype(str)
    df['CustomerID'] = df['CustomerID'].replace('-1', 'Guest')
    
    # Filter out records with invalid UnitPrice (<= 0)
    invalid_price_count = (df['UnitPrice'] <= 0).sum()
    df = df[df['UnitPrice'] > 0]
    print(f"Removed {invalid_price_count:,} transactions with UnitPrice <= 0.")
    
    # Outlier Detection (Capping using 99.9th percentile)
    qty_cap = df['Quantity'].abs().quantile(0.999)
    price_cap = df['UnitPrice'].quantile(0.999)
    print(f"Quantity 99.9th percentile cap: {qty_cap}")
    print(f"UnitPrice 99.9th percentile cap: {price_cap}")
    
    # Keep rows within cap
    original_len = len(df)
    df = df[(df['Quantity'].abs() <= qty_cap) & (df['UnitPrice'] <= price_cap)]
    print(f"Capped outliers: Removed {original_len - len(df):,} transactions.")
    
    # ------------------ PHASE 2: FEATURE ENGINEERING ------------------
    print("\nPhase 2: Feature Engineering...")
    
    # Feature 1: Total Sales (Quantity * UnitPrice)
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    
    # Feature 2: Cost of Goods Sold (COGS) - Simulating at 60% of UnitPrice
    df['COGS'] = df['Quantity'] * (df['UnitPrice'] * 0.60)
    
    # Feature 3: Profit Margin (TotalSales - COGS)
    df['Profit'] = df['TotalSales'] - df['COGS']
    
    # Feature 4: Hour Bin
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['HourBin'] = pd.cut(
        df['Hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
        right=False
    ).astype(str)
    
    # Feature 5: Day of Week & Day Type (Weekend vs Weekday)
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['IsWeekend'] = df['InvoiceDate'].dt.dayofweek.isin([5, 6]).astype(int)
    df['DayType'] = df['IsWeekend'].map({1: 'Weekend', 0: 'Weekday'})
    
    # Feature 6: Order Size Category
    df['OrderSize'] = pd.cut(
        df['Quantity'].abs(),
        bins=[0, 5, 24, 100000],
        labels=['Small', 'Medium', 'Bulk']
    ).astype(str)
    
    # Feature 7: YearMonth (for monthly cohort group sizing)
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    
    # Feature 8: Hemisphere classification (Northern vs Southern)
    southern_countries = ["Australia", "New Zealand", "South Africa", "Brazil"]
    df['Hemisphere'] = np.where(df['Country'].isin(southern_countries), 'Southern', 'Northern')
    
    # Save the cleaned transactions to SQLite database
    print(f"\nSaving {len(df):,} cleaned transactions to database...")
    conn = sqlite3.connect(PROCESSED_DB_PATH)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    
    # ------------------ CUSTOMER SEGMENTATION (RFM & CLV) ------------------
    print("\nBuilding Customer Profiles (RFM & CLV features)...")
    
    # Isolate registered customers (not Guest) for cohort & RFM profiling
    registered_tx = df[(df['CustomerID'] != 'Guest') & (~df['IsCancelled'])].copy()
    
    # Reference date for Recency
    ref_date = registered_tx['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Aggregate to customer level
    customer_profiles = registered_tx.groupby('CustomerID').agg({
        'InvoiceDate': [
            lambda x: (ref_date - x.max()).days,  # Recency
            lambda x: (x.max() - x.min()).days,   # Tenure
            'min'                                 # First purchase (Cohort Month)
        ],
        'InvoiceNo': 'nunique',                   # Frequency
        'TotalSales': 'sum'                       # Monetary
    })
    
    # Flatten column multi-index
    customer_profiles.columns = ['Recency', 'Tenure', 'CohortStartDate', 'Frequency', 'Monetary']
    customer_profiles = customer_profiles.reset_index()
    
    # Convert CohortStartDate to Year-Month string
    customer_profiles['CohortMonth'] = pd.to_datetime(customer_profiles['CohortStartDate']).dt.to_period('M').astype(str)
    customer_profiles = customer_profiles.drop(columns=['CohortStartDate'])
    
    # Calculate Customer Lifetime Value (CLV) simple proxy: Average order size * Frequency
    customer_profiles['CLV'] = customer_profiles['Monetary']
    
    # Score RFM from 1 to 5
    customer_profiles['R_Score'] = pd.qcut(customer_profiles['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    customer_profiles['F_Score'] = pd.qcut(customer_profiles['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    customer_profiles['M_Score'] = pd.qcut(customer_profiles['Monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Combined RFM Score
    customer_profiles['RFM_Segment_Score'] = customer_profiles['R_Score'].astype(str) + customer_profiles['F_Score'].astype(str) + customer_profiles['M_Score'].astype(str)
    
    # Customer Segments based on RFM rules
    def segment_customer(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        avg = (r + f + m) / 3
        if avg >= 4.5:
            return "VIP Champions"
        elif f >= 4 and r >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New/Recent Customers"
        elif r <= 2 and f >= 3:
            return "At Risk / Needs Attention"
        elif r <= 1.5:
            return "Lost / Churned"
        else:
            return "General Active"
            
    customer_profiles['Segment'] = customer_profiles.apply(segment_customer, axis=1)
    
    # Save the customer profiles to SQL
    print(f"Saving {len(customer_profiles):,} customer profiles to database...")
    customer_profiles.to_sql("customers", conn, if_exists="replace", index=False)
    
    # Verify tables inside DB
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"\nTables created in SQLite: {[t[0] for t in tables]}")
    
    conn.close()
    print("\nETL Data Pipeline Completed Successfully! Database is ready.")

if __name__ == "__main__":
    run_data_pipeline()
