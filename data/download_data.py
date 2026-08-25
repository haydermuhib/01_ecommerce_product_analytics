import os
import requests
import pandas as pd

# Define the target paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "online_retail.csv")
URL = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"

def download_dataset():
    print(f"Starting download from: {URL}...")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        
        # Download and write the file in chunks
        with open(CSV_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Successfully downloaded and saved dataset to: {CSV_PATH}")
        
        # Verify the dataset by loading a preview
        df = pd.read_csv(CSV_PATH)
        print("\n--- Dataset Info ---")
        print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print("\nColumns:")
        print(df.dtypes)
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"An error occurred during download/verification: {e}")

if __name__ == "__main__":
    download_dataset()
