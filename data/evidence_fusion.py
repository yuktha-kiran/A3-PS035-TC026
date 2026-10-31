
def fuse_diagnosis(prediction, ranked_faults, evidence, link_results):

    model_probability = float(ranked_faults[0][1])

    report = {
        "predicted_fault": prediction,
        "model_probability": model_probability,
        "suspected_links": [],
        "supporting_evidence": [],
        "warnings": [],
        "diagnosis": ""
    }

    # 1. Collect observable network evidence

    if evidence["connectivity"] == "Unreachable":
        report["supporting_evidence"].append(
            "Network connectivity is unreachable"
        )

    if evidence["latency_ms"] >= 100:
        report["supporting_evidence"].append(
            f"High latency: {evidence['latency_ms']} ms"
        )

    if evidence["packet_loss_pct"] >= 10:
        report["supporting_evidence"].append(
            f"High packet loss: {evidence['packet_loss_pct']}%"
        )

    if evidence["firewall_status"] == "Blocked":
        report["supporting_evidence"].append(
            "Firewall is blocking traffic"
        )

    if evidence["application_status"] == "Failed":
        report["supporting_evidence"].append(
            "Application service has failed"
        )

    # 2. Expected evidence for each fault

    expected_evidence = {
        "Link Failure": "Network connectivity is unreachable",
        "Packet Loss": "High packet loss",
        "High Latency": "High latency",
        "Firewall Block": "Firewall is blocking traffic",
        "Application Failure": "Application service has failed"
    }

    # 3. Evaluate the ML prediction

    if prediction == "Normal":

        if report["supporting_evidence"]:
            report["warnings"].append(
                "Normal prediction conflicts with observed evidence"
            )

            report["diagnosis"] = (
                "Uncertain: possible fault despite Normal prediction"
            )

        else:
            report["diagnosis"] = "No fault detected"

    else:

        expected = expected_evidence.get(prediction)

        # Check whether matching evidence exists

        if expected and not any(
            item.startswith(expected)
            for item in report["supporting_evidence"]
        ):
            report["warnings"].append(
                "Predicted fault lacks matching supporting evidence"
            )

        # Check model probability

        if model_probability < 0.60:
            report["warnings"].append(
                "Model probability is low; diagnosis needs verification"
            )

        # Include suspected links for relevant faults

        if prediction in [
            "Link Failure",
            "Packet Loss",
            "High Latency"
        ]:
            report["suspected_links"] = link_results[:3]

        # Generate final diagnosis

        if report["warnings"]:
            report["diagnosis"] = (
                f"Provisional diagnosis: {prediction}"
            )

        else:
            report["diagnosis"] = (
                f"Evidence-supported diagnosis: {prediction}"
            )

    # 4. Add warning if no abnormal measurements were found

    if not report["supporting_evidence"]:
        if prediction != "Normal":
            report["warnings"].append(
                "No abnormal network measurements were identified"
            )

    return report
