
import random
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from network_simulator import FAULTS, simulate_network

random.seed(100)

# Load trained model
saved = joblib.load("data/trace_model.pkl")
model = saved["model"]
feature_names = saved["features"]

# Load training dataset for numeric imputation values
training_data = pd.read_csv("data/network_dataset.csv")
training_features = pd.get_dummies(
    training_data.drop(columns=["fault"]),
    dtype=int
)

# Create noisy test experiments
test_records = []
actual_labels = []

for fault in FAULTS:
    for _ in range(100):
        evidence = simulate_network(fault)

        # Add random measurement noise
        evidence["latency_ms"] = max(
            1,
            evidence["latency_ms"] + random.randint(-30, 80)
        )

        evidence["packet_loss_pct"] = min(
            100,
            max(
                0,
                evidence["packet_loss_pct"]
                + random.uniform(-3, 8)
            )
        )

        # Occasionally introduce contradictory evidence
        if random.random() < 0.10:
            evidence["connectivity"] = (
                "Unreachable"
                if evidence["connectivity"] == "Reachable"
                else "Reachable"
            )

        if random.random() < 0.10:
            evidence["firewall_status"] = (
                "Blocked"
                if evidence["firewall_status"] == "Allowed"
                else "Allowed"
            )

        # Occasionally remove one evidence field
        if random.random() < 0.15:
            missing_field = random.choice([
                "latency_ms",
                "packet_loss_pct",
                "connectivity",
                "firewall_status",
                "application_status"
            ])
            evidence[missing_field] = None

        actual_labels.append(fault)
        test_records.append(evidence)

# Convert evidence into model features
test_df = pd.DataFrame(test_records)
X_test = pd.get_dummies(
    test_df.drop(columns=["fault"]),
    dtype=int
)

# Align columns with training features
X_test = X_test.reindex(
    columns=feature_names,
    fill_value=0
)

# Impute missing numerical values using training medians
X_test = X_test.fillna(training_features.median())

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(actual_labels, predictions)

print("\nTRACE ROBUSTNESS TEST")
print("=" * 45)

print("Total test experiments:", len(actual_labels))
print(f"Accuracy under noisy evidence: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    actual_labels,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    actual_labels,
    predictions,
    labels=FAULTS
))

print("\nRobustness experiment completed!")
