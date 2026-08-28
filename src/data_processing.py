import sqlite3
import pandas as pd
import numpy as np

def load_data_from_db(db_path):
    """
    Connects to the SQLite database and loads transaction and customer profile tables.
    Casts datetime and boolean values appropriately.
    """
    conn = sqlite3.connect(db_path)
    
    # Load transactions and format column types
    df_tx = pd.read_sql("SELECT * FROM transactions", conn)
    df_tx['InvoiceDate'] = pd.to_datetime(df_tx['InvoiceDate'])
    df_tx['IsCancelled'] = df_tx['IsCancelled'].astype(bool)
    
    # Load customer RFM profiles
    df_cust = pd.read_sql("SELECT * FROM customers", conn)
    
    conn.close()
    return df_tx, df_cust

def calculate_cohort_retention(df_tx):
    """
    Calculates monthly customer cohort retention percentages.
    Filters out Guest transactions and cancellations.
    """
    # Isolate registered customers (excluding guest checkouts)
    df_reg = df_tx[df_tx['CustomerID'] != 'Guest'].copy()
    
    # Get transaction year-month
    df_reg['InvoiceMonth'] = df_reg['InvoiceDate'].dt.to_period('M')
    
    # Identify the cohort month (first purchase month) for each customer
    df_reg['CohortMonth'] = df_reg.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    
    # Group by CohortMonth and InvoiceMonth to count unique active customers
    cohort_group = df_reg.groupby(['CohortMonth', 'InvoiceMonth']).agg(n_customers=('CustomerID', 'nunique')).reset_index()
    
    # Compute the index as the number of months since signup
    cohort_group['CohortIndex'] = (cohort_group['InvoiceMonth'] - cohort_group['CohortMonth']).apply(lambda x: x.n)
    
    # Pivot cohort data into a wide-format grid
    cohort_pivot = cohort_group.pivot(index='CohortMonth', columns='CohortIndex', values='n_customers')
    
    # Divide each month's customer count by the starting cohort size to get percentage
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100
    
    return retention
