
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from network_simulator import FAULTS, simulate_experiment


# Load trained model
saved = joblib.load("data/trace_model.pkl")
model = saved["model"]
feature_names = saved["features"]

# Load training data for missing-value handling
training_data = pd.read_csv("data/network_dataset.csv")

training_features = pd.get_dummies(
    training_data.drop(columns=["fault"]),
    dtype=int
)

medians = training_features.median()

actual = []
predicted = []

# Run 1000 experiments
for i in range(1000):

    fault = FAULTS[i % len(FAULTS)]

    experiment = simulate_experiment(fault)
    evidence = experiment["evidence"]

    input_df = pd.DataFrame([evidence])

    X = pd.get_dummies(input_df, dtype=int)

    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    X = X.fillna(medians)

    prediction = model.predict(X)[0]

    actual.append(fault)
    predicted.append(prediction)


# Calculate metrics
print("\nTRACE - MODEL EVALUATION")
print("=" * 45)

print("Total experiments:", len(actual))

print(
    "Accuracy:",
    f"{accuracy_score(actual, predicted) * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predicted,
        labels=FAULTS,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    actual,
    predicted,
    labels=FAULTS
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=FAULTS
)

display.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("TRACE Fault Classification")
plt.tight_layout()

plt.savefig("data/confusion_matrix.png", dpi=300)

print("\nConfusion matrix saved to data/confusion_matrix.png")

# Save experiment results
results = pd.DataFrame({
    "actual_fault": actual,
    "predicted_fault": predicted
})

results.to_csv(
    "data/evaluation_results.csv",
    index=False
)

print("Results saved to data/evaluation_results.csv")
print("\nEvaluation completed!")
