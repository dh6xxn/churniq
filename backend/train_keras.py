"""Reference TensorFlow/Keras training pipeline for the original project."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Churn_Modelling.csv"
OUT = ROOT / "backend" / "model" / "churn_ann.keras"

def main():
    data = pd.read_csv(DATA).drop(columns=["RowNumber", "CustomerId", "Surname"])
    data["Gender"] = (data["Gender"] == "Male").astype(int)
    data = pd.get_dummies(data, columns=["Geography"], drop_first=True, dtype=int)
    X = data.drop(columns=["Exited"])
    y = data["Exited"]

    # Split before fitting preprocessing to prevent test-set leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1, validation_split=0.1)
    model.save(OUT)
    print(f"Saved Keras model to {OUT}")

if __name__ == "__main__":
    main()
