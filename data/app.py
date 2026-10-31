from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from diagnosis import diagnose
from evidence_fusion import fuse_diagnosis
from network_simulator import FAULTS, LINKS, NODES, simulate_experiment
from topology_analyzer import rank_suspect_links


st.set_page_config(
    page_title="TRACE | Fault localisation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#eaf0f6; --muted:#8da0b3; --line:#243546; --panel:#101c28; --mint:#65e0bd; --blue:#72b7ff; --red:#ff7777; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
    .stApp { background: radial-gradient(ellipse at 75% -12%, #183447 0, #0a131d 42%, #081019 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#0b151f; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--ink); }
    h1,h2,h3 { letter-spacing:-.035em; color:#f2f6fa; }
    .hero { padding:1.2rem 0 1rem; border-bottom:1px solid var(--line); margin-bottom:1.2rem; }
    .eyebrow { color:var(--mint); font:500 .72rem 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; }
    .hero-title { font-size:2.35rem; line-height:1.08; font-weight:800; margin:.3rem 0; }
    .hero-sub { color:var(--muted); font-size:.98rem; }
    .panel { background:linear-gradient(145deg,rgba(19,33,47,.96),rgba(13,24,35,.96)); border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem; }
    .kicker { color:var(--muted); font:500 .68rem 'DM Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }
    .big { color:#f3f7fb; font-size:1.35rem; font-weight:750; margin-top:.25rem; }
    .small { color:var(--muted); font-size:.8rem; margin-top:.15rem; }
    .tag { display:inline-block; border:1px solid #315445; color:var(--mint); background:#10251f; border-radius:99px; padding:.22rem .6rem; font:500 .7rem 'DM Mono',monospace; }
    .evidence { border-left:2px solid var(--mint); background:#11221f; padding:.62rem .8rem; border-radius:0 8px 8px 0; margin:.45rem 0; color:#d7e9e2; }
    .contradiction { border-left:2px solid #e5ad60; background:#261f15; padding:.62rem .8rem; border-radius:0 8px 8px 0; margin:.45rem 0; color:#f0dfc2; }
    .muted { color:var(--muted); }
    div[data-testid="stMetric"] { background:rgba(16,28,40,.9); border:1px solid var(--line); padding:.75rem .9rem; border-radius:12px; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stMetricValue"] { color:#f2f6fa; }
    .stButton button[kind="primary"] { background:#62d9b5; color:#07140f; border:0; font-weight:800; border-radius:9px; }
    .stButton button { border-radius:9px; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    saved = joblib.load(BASE_DIR / "trace_model.pkl")
    model = saved["model"]
    features = saved["features"]
    training = pd.read_csv(BASE_DIR / "network_dataset.csv")
    encoded = pd.get_dummies(training.drop(columns=["fault"]), dtype=int)
    return model, features, encoded.median()


model, feature_names, medians = load_assets()


def run_diagnosis(scenario):
    experiment = simulate_experiment(scenario)
    evidence = experiment["evidence"]
    x = pd.get_dummies(pd.DataFrame([evidence]), dtype=int)
    x = x.reindex(columns=feature_names, fill_value=0).fillna(medians)
    prediction = model.predict(x)[0]
    probs = model.predict_proba(x)[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda pair: pair[1], reverse=True)
    links = rank_suspect_links(experiment["path_observations"])
    report = fuse_diagnosis(prediction, ranked, evidence, links)
    rule_results = diagnose(evidence)
    # The deterministic rule baseline only fires on explicit threshold matches.
    rule_fault = rule_results[0]["fault"] if rule_results else "Normal"
    support = list(report["supporting_evidence"])
    contradictions = []
    expected = {
        "Link Failure": evidence["connectivity"] == "Unreachable",
        "Packet Loss": evidence["packet_loss_pct"] >= 10,
        "High Latency": evidence["latency_ms"] >= 100,
        "Firewall Block": evidence["firewall_status"] == "Blocked",
        "Application Failure": evidence["application_status"] == "Failed",
    }
    if prediction != "Normal" and not expected.get(prediction, False):
        contradictions.append(f"The model's {prediction} prediction lacks its expected direct indicator.")
    if prediction == "Normal" and support:
        contradictions.append("The model predicts Normal despite abnormal measurements.")
    if prediction != rule_fault:
        contradictions.append(f"Rule baseline predicts {rule_fault}; Random Forest predicts {prediction}.")
    if scenario != "Normal" and prediction == "Normal":
        contradictions.append("Injected scenario and predicted class disagree; inspect the evidence before acting.")
    return {
        "scenario": scenario,
        "experiment": experiment,
        "evidence": evidence,
        "prediction": prediction,
        "ranked": ranked,
        "links": links,
        "report": report,
        "rule_fault": rule_fault,
        "support": support,
        "contradictions": contradictions,
    }


if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


with st.sidebar:
    st.markdown("### ◈ TRACE")
    st.caption("Topology-aware root-cause analysis")
    st.markdown("---")
    scenario = st.selectbox("Fault injection scenario", FAULTS, index=0)
    run_clicked = st.button("Run diagnosis", type="primary", use_container_width=True)
    if run_clicked:
        result = run_diagnosis(scenario)
        st.session_state.latest_result = result
        st.session_state.history.insert(0, {
            "Scenario": scenario,
            "Model result": result["prediction"],
            "Model probability": f"{result['ranked'][0][1]:.1%}",
            "Location": "-".join(result["experiment"]["fault_location"]) if isinstance(result["experiment"]["fault_location"], tuple) else result["experiment"]["fault_location"] or "—",
        })
    st.markdown("---")
    st.markdown("**Testbed** · 5 nodes · 5 links")
    st.markdown("**Engines** · Rules + Random Forest")
    st.markdown("**Evidence** · Measurements + topology")
    st.caption("Simulation only · no live network telemetry")


st.markdown(
    '<div class="hero"><div class="eyebrow">AIORI-3 · PS-035</div>'
    '<div class="hero-title">TRACE <span style="color:#65e0bd">/</span> Fault intelligence</div>'
    '<div class="hero-sub">Correlate network symptoms with topology to explain where a fault may have started.</div></div>',
    unsafe_allow_html=True,
)
st.info("This prototype generates controlled synthetic experiments. Results demonstrate the diagnosis workflow; they do not represent a live network or calibrated operational confidence.")

overview_tab, diagnosis_tab, evidence_tab, evaluation_tab = st.tabs(["Network overview", "Diagnosis", "Evidence explorer", "Evaluation"])

with overview_tab:
    top = st.columns([1.4, 1, 1, 1])
    top[0].markdown("### Five-node testbed")
    top[1].metric("Nodes", len(NODES))
    top[2].metric("Links", len(LINKS))
    top[3].metric("Fault scenarios", len(FAULTS) - 1)
    st.caption("Reference topology used by the simulator and BFS path analyser.")
    link_scores = {}
    if st.session_state.latest_result:
        link_scores = {row["link"]: row for row in st.session_state.latest_result["links"]}
    graph = 'graph TRACE { layout=neato; bgcolor="transparent"; node [shape=circle,style=filled,fillcolor="#17364a",fontcolor="#eaf0f6",color="#65e0bd",penwidth=2,width=.72,height=.72,fontname="Arial"]; edge [color="#476073",penwidth=2];'
    for source, destination in LINKS:
        key = "-".join(sorted((source, destination)))
        if key in link_scores:
            graph += f' "{source}" -- "{destination}" [color="#ff7777", penwidth=4, label="suspect"];'
        else:
            graph += f' "{source}" -- "{destination}";'
    graph += " }"
    st.graphviz_chart(graph, use_container_width=True)
    if link_scores:
        st.caption("Red links are ranked as suspect from path observations. Compare them with the injected location in the Diagnosis tab.")
    st.markdown("#### Diagnosis pipeline")
    stages = st.columns(5)
    for col, (number, label, detail) in zip(stages, [
        ("01", "Inject", "Controlled scenario"), ("02", "Observe", "Synthetic telemetry"),
        ("03", "Correlate", "Paths + symptoms"), ("04", "Rank", "Rules + Random Forest"),
        ("05", "Explain", "Evidence + caveats"),
    ]):
        with col:
            st.markdown(f'<div class="panel"><div class="eyebrow">{number}</div><b>{label}</b><div class="small">{detail}</div></div>', unsafe_allow_html=True)

with diagnosis_tab:
    result = st.session_state.latest_result
    if result is None:
        st.markdown("### Ready for a controlled run")
        st.write("Choose a scenario in the left panel and select **Run diagnosis**. TRACE will generate measurements, rank candidate causes, and explain which evidence supports the result.")
        st.caption("No experiment has been run in this session yet.")
    else:
        report, evidence = result["report"], result["evidence"]
        st.markdown("### Root-cause assessment")
        headline = st.columns([2, 1, 1, 1])
        headline[0].markdown(f'<div class="panel"><div class="kicker">{report["diagnosis"]}</div><div class="big">{result["prediction"]}</div><div class="small">Scenario injected: {result["scenario"]}</div></div>', unsafe_allow_html=True)
        headline[1].metric("Model probability", f"{result['ranked'][0][1]:.1%}")
        headline[2].metric("Rule baseline", result["rule_fault"])
        loc = result["experiment"]["fault_location"]
        headline[3].metric("Injected location", "-".join(loc) if isinstance(loc, tuple) else loc or "None")
        st.caption("Model probability is the Random Forest output for this synthetic sample. It is not calibrated diagnostic confidence.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Reachability", evidence["connectivity"])
        metric_cols[1].metric("Packet loss", f"{evidence['packet_loss_pct']}%")
        metric_cols[2].metric("Latency", f"{evidence['latency_ms']} ms")
        metric_cols[3].metric("Service", evidence["application_status"])
        left, right = st.columns([1.15, .85])
        with left:
            st.markdown("#### Candidate ranking")
            ranked_df = pd.DataFrame(result["ranked"], columns=["Candidate", "Model probability"])
            ranked_df["Model probability"] = (ranked_df["Model probability"] * 100).round(1)
            st.bar_chart(ranked_df.set_index("Candidate"), horizontal=True)
            st.dataframe(ranked_df.rename(columns={"Model probability": "Probability (%)"}), hide_index=True, use_container_width=True)
        with right:
            st.markdown("#### Link localisation")
            if result["links"]:
                st.dataframe(pd.DataFrame(result["links"]), hide_index=True, use_container_width=True)
                top_link = result["links"][0]["link"]
                st.caption(f"Highest topology suspicion: **{top_link}**. The injected location is shown separately for comparison.")
            else:
                st.success("No failed or degraded path evidence in this run.")
        st.markdown("#### Why TRACE reached this result")
        explain_cols = st.columns(2)
        with explain_cols[0]:
            st.markdown("**Supporting observations**")
            if result["support"]:
                for item in result["support"]:
                    st.markdown(f'<div class="evidence">✓ &nbsp;{item}</div>', unsafe_allow_html=True)
            else:
                st.caption("No thresholded abnormal measurement was found.")
        with explain_cols[1]:
            st.markdown("**Conflicts and uncertainty**")
            if result["contradictions"] or report["warnings"]:
                for item in dict.fromkeys(result["contradictions"] + report["warnings"]):
                    st.markdown(f'<div class="contradiction">! &nbsp;{item}</div>', unsafe_allow_html=True)
            else:
                st.caption("No explicit conflict was identified by the current checks.")

with evidence_tab:
    result = st.session_state.latest_result
    if result is None:
        st.caption("Run a scenario to inspect its generated measurements and path observations.")
    else:
        st.markdown("### Collected evidence")
        e = result["evidence"]
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Network measurements**")
        c1.dataframe(pd.DataFrame([{"Signal": k, "Observed value": v} for k, v in e.items() if k not in ("fault", "nodes", "links")]), hide_index=True, use_container_width=True)
        c2.markdown("**Link probes**")
        c2.dataframe(pd.DataFrame(result["experiment"]["path_observations"]), hide_index=True, use_container_width=True)
        c3.markdown("**Node health**")
        c3.dataframe(pd.DataFrame(result["experiment"]["node_observations"]), hide_index=True, use_container_width=True)
        st.caption("The simulator creates generated observations for each physical link and node. Syslog ingestion and live ICMP probing are not implemented in this prototype; production use would need timestamped collectors for both.")

with evaluation_tab:
    st.markdown("### Model evaluation snapshot")
    eval_path = BASE_DIR / "evaluation_results.csv"
    if eval_path.exists():
        outcomes = pd.read_csv(eval_path)
        accuracy = (outcomes["actual_fault"] == outcomes["predicted_fault"]).mean()
        top1, count, scope = st.columns(3)
        top1.metric("Top-1 accuracy", f"{accuracy:.1%}")
        count.metric("Recorded experiments", len(outcomes))
        scope.metric("Evaluation type", "Synthetic simulation")
        st.caption("This CSV is generated from simulator scenarios. It is a prototype evaluation, not an independent field validation. Top-3 accuracy is not recorded in the current evaluation file.")
        by_class = (outcomes.assign(correct=outcomes["actual_fault"] == outcomes["predicted_fault"])
                    .groupby("actual_fault", as_index=False)["correct"].mean()
                    .rename(columns={"actual_fault": "Scenario", "correct": "Top-1 accuracy"}))
        by_class["Top-1 accuracy"] *= 100
        st.markdown("#### Accuracy by injected scenario")
        st.bar_chart(by_class.set_index("Scenario"))
        with st.expander("Recorded predictions"):
            st.dataframe(outcomes, hide_index=True, use_container_width=True)
    else:
        st.warning("No evaluation_results.csv found. Run data/evaluation.py to generate the evaluation snapshot.")

if st.session_state.history:
    with st.expander("Recent experiment history", expanded=False):
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
