import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

def train_model():
    df = pd.read_csv("data_processed.csv")

    X = df.drop(columns=["Status"])  # Ganti jika target berbeda
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("✅ Model trained and logged with MLflow (autolog).")

if __name__ == "__main__":
    train_model()
