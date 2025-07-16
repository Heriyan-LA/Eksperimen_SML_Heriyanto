
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(
    data_path='data_processed.csv',
    target_column='Status',          # Ganti dengan kolom target sesuai data
    test_size=0.2,
    random_state=42
):
    # Load preprocessed data
    df = pd.read_csv(data_path)

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Inisialisasi dan latih model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    print("✅ Akurasi:", accuracy_score(y_test, y_pred))
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

# Eksekusi saat dijalankan langsung
if __name__ == '__main__':
    train_model()
