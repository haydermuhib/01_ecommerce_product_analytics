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

def map_season_hemisphere(row):
    """
    Identifies the correct season for a transaction row, taking into account
    the precomputed Hemisphere column (e.g. Southern vs Northern).
    """
    month = row['InvoiceDate'].month
    hemisphere = row['Hemisphere']
    
    if hemisphere == 'Southern':
        if month in [12, 1, 2]:
            return "Summer"
        elif month in [3, 4, 5]:
            return "Autumn"
        elif month in [6, 7, 8]:
            return "Winter"
        else:
            return "Spring"
    else:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"


def calculate_seasonal_product_data(df_filtered, selected_season, metric_direction, num_products):
    """
    Calculates product performance metrics filtered by hemisphere-aware seasons.
    Returns grouped seasonal dataset and the sliced top/bottom performers dataframe.
    """
    df_filtered_copy = df_filtered.copy()
    df_filtered_copy['Season'] = df_filtered_copy.apply(map_season_hemisphere, axis=1)
    
    # Filter for active seasonal records
    df_season = df_filtered_copy[(df_filtered_copy['Season'] == selected_season) & (~df_filtered_copy['IsCancelled'])]
    
    # Group by product description
    df_season_grouped = df_season.groupby('Description').agg(
        Revenue=('TotalSales', 'sum'),
        Units=('Quantity', 'sum')
    ).reset_index()
    
    # Select performance direction
    if "Bottom" in metric_direction:
        df_season_grouped = df_season_grouped[df_season_grouped['Revenue'] > 0]
        df_season_top = df_season_grouped.sort_values(by='Revenue', ascending=True).head(num_products)
    else:
        df_season_top = df_season_grouped.sort_values(by='Revenue', ascending=False).head(num_products)
        
    return df_season, df_season_top
