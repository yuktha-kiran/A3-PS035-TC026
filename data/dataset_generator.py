
import random
import pandas as pd

from network_simulator import FAULTS, simulate_network

# Make experiments reproducible
random.seed(42)

dataset = []

# Generate 500 experiments for each scenario
for fault in FAULTS:
    for _ in range(500):
        evidence = simulate_network(fault)
        dataset.append(evidence)

# Convert experiments into a table
df = pd.DataFrame(dataset)

# Save dataset
df.to_csv("data/network_dataset.csv", index=False)

print("TRACE Dataset Generated!")
print("=" * 40)

print("Total experiments:", len(df))
print("\nClass distribution:")
print(df["fault"].value_counts())

print("\nDataset saved to data/network_dataset.csv")
