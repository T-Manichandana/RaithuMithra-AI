
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Find the project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "crop_model.joblib"

# 2. Check whether the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place Crop_recommendation.csv inside the data folder."
    )

# 3. Load the dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# 4. Select input features and target
features = [
    "n", "p", "k", "temperature",
    "humidity", "ph", "rainfall"
]
target = "label"

required_columns = features + [target]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {list(df.columns)}"
    )

df = df[required_columns].dropna()

X = df[features]
y = df[target]

# 5. Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("Training the crop recommendation model...")
model.fit(X_train, y_train)

# 7. Evaluate the model on unseen test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions))

# 8. Save the trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully at:\n{MODEL_PATH}")
