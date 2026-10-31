# AIORI-3-HACKATHON-PROJECTS
This repository will contain the code and resources for the AIORI-3 hackathon that will be developed during the AIORI-3
## Team Video

**Project:** TRACE — Topology-aware Root-cause Analysis with Correlated Evidence

**Problem Statement:** PS-035 — Evidence-Based Operator Network Fault Localisation

**Hackathon:** AIORI-3 Online Hackathon 2026

**Team Code:** A3-PS035-TC026

**Video Presentation:** [Watch our team introduction and project presentation](https://youtu.be/FABaUDOUCnQ)

**Video Demonstration:** (https://youtu.be/EXBcSTZXVTs)

## Run the TRACE dashboard

The dashboard runs a controlled, synthetic five-node network experiment. It does not monitor a live network.

From this repository directory, install the data dependencies and start Streamlit:

```powershell
python -m pip install -r data/requirements.txt
python -m streamlit run data/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Choose a fault scenario in the sidebar and select **Run diagnosis**. The dashboard shows the Random Forest ranking, deterministic rule baseline, topology-based suspect links, supporting and conflicting evidence, and the available synthetic evaluation results.

The saved model and dataset are loaded relative to `data/app.py`, so the dashboard can be launched from the repository root without changing directories.
