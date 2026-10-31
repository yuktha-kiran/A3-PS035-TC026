
def diagnose(evidence):
    """Rank likely causes using diagnostic evidence."""

    scores = {
        "Link Failure": 0,
        "Packet Loss": 0,
        "High Latency": 0,
        "Firewall Block": 0,
        "Application Failure": 0,
    }

    reasons = {
        fault: [] for fault in scores
    }

    if evidence["connectivity"] == "Unreachable":
        scores["Link Failure"] += 3
        reasons["Link Failure"].append(
            "Connectivity is unreachable"
        )

    if evidence["packet_loss_pct"] >= 10:
        scores["Packet Loss"] += 3
        reasons["Packet Loss"].append(
            f"Packet loss is {evidence['packet_loss_pct']}%"
        )

    if evidence["latency_ms"] >= 100:
        scores["High Latency"] += 3
        reasons["High Latency"].append(
            f"Latency is {evidence['latency_ms']} ms"
        )

    if evidence["firewall_status"] == "Blocked":
        scores["Firewall Block"] += 3
        reasons["Firewall Block"].append(
            "Firewall reports blocked traffic"
        )

    if evidence["application_status"] == "Failed":
        scores["Application Failure"] += 3
        reasons["Application Failure"].append(
            "Application health check failed"
        )

    ranking = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for fault, score in ranking:
        if score == 0:
            continue

        results.append({
            "fault": fault,
            "score": score,
            "evidence": reasons[fault],
        })

    return results


if __name__ == "__main__":
    sample = {
        "connectivity": "Reachable",
        "packet_loss_pct": 32.5,
        "latency_ms": 24,
        "firewall_status": "Allowed",
        "application_status": "Healthy",
    }

    print("TRACE Diagnosis Results")
    print("=" * 30)

    for result in diagnose(sample):
        print(f"\nRoot cause: {result['fault']}")
        print(f"Evidence score: {result['score']}")
        print("Supporting evidence:", result["evidence"])
