
import random

NODES = ["A", "B", "C", "D", "E"]

LINKS = [
    ("A", "B"),
    ("B", "C"),
    ("B", "D"),
    ("C", "E"),
    ("D", "E"),
]

FAULTS = [
    "Normal",
    "Link Failure",
    "Packet Loss",
    "High Latency",
    "Firewall Block",
    "Application Failure",
]


def simulate_network(fault="Normal"):
    """Generate simulated network diagnostic evidence."""

    evidence = {
        "fault": fault,
        "nodes": len(NODES),
        "links": len(LINKS),
        "latency_ms": random.randint(10, 30),
        "packet_loss_pct": round(random.uniform(0, 1), 2),
        "connectivity": "Reachable",
        "firewall_status": "Allowed",
        "application_status": "Healthy",
    }

    if fault == "Link Failure":
        evidence["connectivity"] = "Unreachable"

    elif fault == "Packet Loss":
        evidence["packet_loss_pct"] = round(
            random.uniform(20, 45), 2
        )

    elif fault == "High Latency":
        evidence["latency_ms"] = random.randint(200, 500)

    elif fault == "Firewall Block":
        evidence["firewall_status"] = "Blocked"
        evidence["connectivity"] = "Blocked"

    elif fault == "Application Failure":
        evidence["application_status"] = "Failed"

    return evidence


def simulate_experiment(fault="Normal"):
    """
    Generate one correlated network experiment.

    Link Failure, Packet Loss and High Latency:
        Inject a fault into a randomly selected link.

    Firewall Block and Application Failure:
        Inject a fault at a randomly selected node.

    Normal:
        No fault is injected.
    """

    evidence = simulate_network(fault)

    fault_location = None
    fault_location_type = None

    # Randomly select a faulty link
    if fault in ["Link Failure", "Packet Loss", "High Latency"]:

        fault_location = random.choice(LINKS)
        fault_location_type = "link"

    # Randomly select a faulty node
    elif fault in ["Firewall Block", "Application Failure"]:

        fault_location = random.choice(NODES[1:])
        fault_location_type = "node"

    # Generate observations for every physical link
    observations = []

    for source, destination in LINKS:

        is_faulty_link = (
            fault_location_type == "link"
            and fault_location == (source, destination)
        )

        status = "healthy"
        severity = 0.0

        if is_faulty_link:

            if fault == "Link Failure":
                status = "failed"
                severity = 1.0

            elif fault in ["Packet Loss", "High Latency"]:
                status = "degraded"
                severity = 2.0

        observation = {
            "source": source,
            "destination": destination,
            "status": status,
            "severity": severity,
            "latency_ms": (
                evidence["latency_ms"]
                if is_faulty_link and fault == "High Latency"
                else random.randint(10, 30)
            ),
            "packet_loss_pct": (
                evidence["packet_loss_pct"]
                if is_faulty_link and fault == "Packet Loss"
                else round(random.uniform(0, 1), 2)
            )
        }

        observations.append(observation)

    # Additional node-level diagnostic evidence
    node_observations = []

    for node in NODES:

        node_status = "healthy"

        if node == fault_location and fault_location_type == "node":
            node_status = (
                "blocked"
                if fault == "Firewall Block"
                else "service_failed"
            )

        node_observations.append({
            "node": node,
            "status": node_status
        })

    return {
        "evidence": evidence,
        "fault_location": fault_location,
        "fault_location_type": fault_location_type,
        "path_observations": observations,
        "node_observations": node_observations
    }


if __name__ == "__main__":

    print("TRACE - Location-Aware Network Simulation")
    print("=" * 50)

    print("Nodes:", NODES)
    print("Links:", LINKS)

    for fault in FAULTS:

        experiment = simulate_experiment(fault)

        print(f"\nScenario: {fault}")
        print("Fault location:", experiment["fault_location"])
        print("Location type:", experiment["fault_location_type"])

        print("Network evidence:")
        print(experiment["evidence"])

        print("Link observations:")
        for observation in experiment["path_observations"]:
            print(observation)

        print("Node observations:")
        print(experiment["node_observations"])
