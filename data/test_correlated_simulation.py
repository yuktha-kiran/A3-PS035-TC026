from network_simulator import simulate_experiment

experiment = simulate_experiment("Link Failure")

print("\nTRACE - Correlated Network Experiment")
print("=" * 45)

print("\n1. NETWORK EVIDENCE")
for key, value in experiment["evidence"].items():
    print(f"{key}: {value}")

print("\n2. INJECTED FAULT LOCATION")
print(experiment["fault_location"])

print("\n3. PATH OBSERVATIONS")
for observation in experiment["path_observations"]:
    print(observation)