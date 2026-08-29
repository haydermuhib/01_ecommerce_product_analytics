import os
import requests
import pandas as pd

# Define paths dynamically relative to the script's location (data/)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CSV_PATH = os.path.join(RAW_DATA_DIR, "online_retail.csv")
URL = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"

def download_dataset():
    # Ensure data/raw/ directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    print(f"Starting dataset download from: {URL}...")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        
        with open(CSV_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Successfully downloaded and saved raw dataset to: {CSV_PATH}")
        
        # Verify the download structure
        df = pd.read_csv(CSV_PATH)
        print(f"Verification check: loaded {df.shape[0]:,} rows and {df.shape[1]} columns successfully.")
        
    except Exception as e:
        print(f"An error occurred during dataset download or verification: {e}")

if __name__ == "__main__":
    download_dataset()
