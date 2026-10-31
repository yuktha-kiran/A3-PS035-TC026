
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/network_dataset.csv")

# Convert categorical evidence into numeric features
X = pd.get_dummies(
    df.drop(columns=["fault"]),
    dtype=int
)

y = df["fault"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predict test results
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("TRACE - Random Forest Evaluation")
print("=" * 40)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and feature columns
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist()
    },
    "data/trace_model.pkl"
)

print("\nModel saved to data/trace_model.pkl")
