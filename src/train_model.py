import os
import pickle
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Define database and output model paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DB_PATH = os.path.join(PROJECT_DIR, "data", "processed", "ecommerce.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

def train_and_save_model():
    print("Connecting to database and pulling customer profile records...")
    conn = sqlite3.connect(DB_PATH)
    df_cust = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()

    print(f"Loaded {len(df_cust):,} customer profiles.")

    # Define target label: Churn = 1 (if customer has not made a purchase in 90+ days)
    # Recency is the number of days since last purchase
    df_cust["IsChurned"] = (df_cust["Recency"] > 90).astype(int)

    # Feature selection
    # We use Recency's underlying components (Frequency, Monetary, Tenure) to prevent data leakage
    features = ["Frequency", "Monetary", "Tenure"]
    X = df_cust[features]
    y = df_cust["IsChurned"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    print(f"Churn rate in dataset: {y.mean() * 100:.2f}%")

    # Create a pipeline combining feature scaling and random forest classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])

    # Fit the model
    print("Fitting Random Forest classification pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation Metrics:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Ensure models directory exists and serialize model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
        
    print(f"\nSuccessfully saved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
