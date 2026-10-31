
import streamlit as st
import pandas as pd
import joblib

from network_simulator import FAULTS, simulate_experiment
from topology_analyzer import rank_suspect_links
from evidence_fusion import fuse_diagnosis


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TRACE | Network Fault Localisation",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 TRACE")
st.subheader("Evidence-Based Network Fault Localisation")

st.caption(
    "Topology-aware Root-cause Analysis with Correlated Evidence"
)

st.info(
    "Prototype mode: network measurements are simulated. "
    "This dashboard does not monitor a live network."
)


# --------------------------------------------------
# 2. LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    saved = joblib.load("data/trace_model.pkl")

    model = saved["model"]
    feature_names = saved["features"]

    training_data = pd.read_csv("data/network_dataset.csv")

    training_features = pd.get_dummies(
        training_data.drop(columns=["fault"]),
        dtype=int
    )

    medians = training_features.median()

    return model, feature_names, medians


model, feature_names, medians = load_model()


# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------

st.sidebar.title("TRACE Controls")

scenario = st.sidebar.selectbox(
    "Select Network Scenario",
    FAULTS
)

run_button = st.sidebar.button(
    "Run Network Diagnosis",
    type="primary",
    use_container_width=True
)

st.sidebar.divider()

st.sidebar.markdown("### Network Configuration")
st.sidebar.write("Network Nodes: 5")
st.sidebar.write("Network Links: 5")
st.sidebar.write("ML Algorithm: Random Forest")
st.sidebar.write("Topology Analysis: BFS-based")


# --------------------------------------------------
# 4. EXPERIMENT HISTORY
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# 5. RUN DIAGNOSIS
# --------------------------------------------------

if run_button:

    # Generate correlated experiment
    experiment = simulate_experiment(scenario)

    evidence = experiment["evidence"]
    observations = experiment["path_observations"]

    # Prepare ML input
    input_df = pd.DataFrame([evidence])

    X = pd.get_dummies(
        input_df,
        dtype=int
    )

    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    X = X.fillna(medians)

    # ML prediction
    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    ranked_faults = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    # Topology analysis
    link_results = rank_suspect_links(observations)

    # Evidence fusion
    final_report = fuse_diagnosis(
        prediction,
        ranked_faults,
        evidence,
        link_results
    )

    # Save latest experiment
    st.session_state.latest_result = {
        "scenario": scenario,
        "prediction": prediction,
        "probability": float(ranked_faults[0][1]),
        "evidence": evidence,
        "experiment": experiment,
        "links": link_results,
        "report": final_report
    }

    # Save experiment history
    st.session_state.history.insert(
        0,
        {
            "Scenario": scenario,
            "Prediction": prediction,
            "Probability": f"{ranked_faults[0][1]:.2%}"
        }
    )


# --------------------------------------------------
# 6. DISPLAY DASHBOARD
# --------------------------------------------------

if "latest_result" in st.session_state:

    result = st.session_state.latest_result

    evidence = result["evidence"]
    prediction = result["prediction"]
    report = result["report"]
    link_results = result["links"]
    experiment = result["experiment"]

    st.success("Network diagnosis completed!")

    # ----------------------------------------------
    # FINAL DIAGNOSIS
    # ----------------------------------------------

    st.markdown("## Final Diagnosis")

    st.subheader(report["diagnosis"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Detected Fault",
        prediction
    )

    col2.metric(
        "Model Probability",
        f"{report['model_probability']:.2%}"
    )

    col3.metric(
        "Suspected Links",
        len(report["suspected_links"])
    )

    st.divider()

    # ----------------------------------------------
    # NETWORK METRICS
    # ----------------------------------------------

    st.markdown("## Network Monitoring")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latency",
        f"{evidence['latency_ms']} ms"
    )

    c2.metric(
        "Packet Loss",
        f"{evidence['packet_loss_pct']}%"
    )

    c3.metric(
        "Connectivity",
        evidence["connectivity"]
    )

    c4.metric(
        "Network Nodes",
        evidence["nodes"]
    )

    c1, c2, c3 = st.columns(3)

    c1.write(
        f"**Firewall:** {evidence['firewall_status']}"
    )

    c2.write(
        f"**Application:** {evidence['application_status']}"
    )

    c3.write(
        f"**Injected Fault:** {experiment['fault_location']}"
    )

    st.divider()

    # ----------------------------------------------
    # NETWORK TOPOLOGY
    # ----------------------------------------------

    st.markdown("## Network Topology")

    st.caption(
        "Red links indicate the highest-ranked suspected link. "
        "Other links are displayed in grey."
    )

    suspected = set()

    for link in report["suspected_links"]:
        suspected.add(link["link"])

    graph = """
    graph TRACE {
        layout=neato;
        bgcolor="transparent";
        node [
            shape=circle,
            style=filled,
            fillcolor="#24496B",
            fontcolor="white",
            color="#8AA5BE",
            penwidth=2,
            width=0.65,
            height=0.65
        ];

        edge [
            color="#8494A7",
            penwidth=2
        ];

        A -- B;
        B -- C;
        B -- D;
        C -- E;
        D -- E;
    """

    edges = [
        ("A", "B"),
        ("B", "C"),
        ("B", "D"),
        ("C", "E"),
        ("D", "E")
    ]

    for source, destination in edges:

        link_name = "-".join(sorted([source, destination]))

        if link_name in suspected:
            graph += (
                f'\n"{source}" -- "{destination}" '
                '[color="#E05252", penwidth=5];'
            )

    graph += "\n}"

    st.graphviz_chart(
        graph,
        use_container_width=True
    )

    if report["suspected_links"]:
        st.warning(
            "Potential network fault location identified."
        )
    else:
        st.info(
            "No suspected faulty links identified in this experiment."
        )

    st.divider()

    # ----------------------------------------------
    # ML PREDICTION DETAILS
    # ----------------------------------------------

    st.markdown("## Machine Learning Analysis")

    probability_df = pd.DataFrame(
        ranked_faults,
        columns=["Fault", "Probability"]
    )

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    )

    probability_df = probability_df.set_index("Fault")

    st.bar_chart(
        probability_df,
        y="Probability"
    )

    # ----------------------------------------------
    # TOPOLOGY RESULTS
    # ----------------------------------------------

    st.markdown("## Topology Analysis")

    if link_results:

        link_df = pd.DataFrame(link_results)

        st.dataframe(
            link_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No failed paths detected.")

    # ----------------------------------------------
    # SUPPORTING EVIDENCE
    # ----------------------------------------------

    st.markdown("## Supporting Evidence")

    if report["supporting_evidence"]:

        for item in report["supporting_evidence"]:
            st.write("✅", item)

    else:
        st.write("No abnormal evidence identified.")

    # ----------------------------------------------
    # WARNINGS
    # ----------------------------------------------

    st.markdown("## Diagnostic Warnings")

    if report["warnings"]:

        for warning in report["warnings"]:
            st.warning(warning)

    else:
        st.success("No diagnostic warnings.")

    # ----------------------------------------------
    # RAW EVIDENCE
    # ----------------------------------------------

    with st.expander("View Raw Diagnostic Evidence"):

        st.json(evidence)

        st.markdown("### Path Observations")

        st.json(experiment["path_observations"])


else:

    st.markdown("## Welcome to TRACE")

    st.write(
        "Select a network scenario from the sidebar "
        "and click Run Network Diagnosis to begin."
    )

    st.info(
        "Your diagnostic results, network topology "
        "and supporting evidence will appear here."
    )


# --------------------------------------------------
# 7. EXPERIMENT HISTORY
# --------------------------------------------------

st.divider()

st.markdown("## Experiment History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Experiment History"):
        st.session_state.history = []
        st.rerun()

else:

    st.caption("No experiments have been recorded yet.")
