
import joblib
import pandas as pd

from network_simulator import FAULTS, simulate_experiment
from evidence_fusion import fuse_diagnosis
from topology_analyzer import rank_suspect_links


# 1. Load trained Random Forest model
saved = joblib.load("data/trace_model.pkl")

model = saved["model"]
feature_names = saved["features"]


# 2. Load training data for missing-value imputation
training_data = pd.read_csv("data/network_dataset.csv")

training_features = pd.get_dummies(
    training_data.drop(columns=["fault"]),
    dtype=int
)

medians = training_features.median()


# 3. Display available scenarios
print("\nTRACE - Integrated Diagnosis System")
print("=" * 45)

print("\nAvailable scenarios:")

for i, fault in enumerate(FAULTS, start=1):
    print(f"{i}. {fault}")


choice = int(input("\nChoose a scenario (1-6): "))

if choice < 1 or choice > len(FAULTS):
    raise ValueError("Please choose a number between 1 and 6.")

scenario = FAULTS[choice - 1]


# 4. Generate correlated network experiment
experiment = simulate_experiment(scenario)

evidence = experiment["evidence"]
observations = experiment["path_observations"]


# 5. Prepare model input
input_df = pd.DataFrame([evidence])

X = pd.get_dummies(input_df, dtype=int)

X = X.reindex(
    columns=feature_names,
    fill_value=0
)

X = X.fillna(medians)


# 6. Predict network fault
prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

ranked_faults = sorted(
    zip(model.classes_, probabilities),
    key=lambda item: item[1],
    reverse=True
)


# 7. Machine Learning Diagnosis
print("\n1. MACHINE LEARNING DIAGNOSIS")
print("-" * 35)

print("Predicted fault:", prediction)

for fault, probability in ranked_faults[:3]:
    print(f"{fault}: {probability:.2%}")

print("Injected fault location:", experiment["fault_location"])


# 8. Topology Analysis
print("\n2. TOPOLOGY ANALYSIS")
print("-" * 35)

link_results = rank_suspect_links(observations)

if link_results:
    for result in link_results:
        print(
            f"Link {result['link']} | "
            f"Suspicion Score: {result['suspicion_score']} | "
            f"Failed Support: {result['failed_path_support']} | "
            f"Healthy Support: {result['healthy_path_support']}"
        )
else:
    print("No failed paths detected.")


# 9. Final Evidence Fusion
print("\n3. FINAL EVIDENCE FUSION")
print("-" * 35)

final_report = fuse_diagnosis(
    prediction,
    ranked_faults,
    evidence,
    link_results
)

print("\nFinal Diagnosis:")
print(final_report["diagnosis"])

print("\nModel Probability:")
print(f"{final_report['model_probability']:.2%}")


# 10. Suspected Links
print("\nSuspected Links:")

if final_report["suspected_links"]:
    for link in final_report["suspected_links"]:
        print(
            f"Link {link['link']} | "
            f"Suspicion Score: {link['suspicion_score']} | "
            f"Failed Support: {link['failed_path_support']} | "
            f"Healthy Support: {link['healthy_path_support']}"
        )
else:
    print("No suspicious links identified.")


# 11. Supporting Evidence
print("\nSupporting Evidence:")

if final_report["supporting_evidence"]:
    for item in final_report["supporting_evidence"]:
        print("-", item)
else:
    print("No abnormal evidence identified.")


# 12. Diagnostic Warnings
print("\nWarnings:")

if final_report["warnings"]:
    for warning in final_report["warnings"]:
        print("-", warning)
else:
    print("No warnings.")


# 13. Raw Diagnostic Evidence
print("\n4. RAW DIAGNOSTIC EVIDENCE")
print("-" * 35)

for key, value in evidence.items():
    print(f"{key}: {value}")


print("\nTRACE integrated experiment completed!")
