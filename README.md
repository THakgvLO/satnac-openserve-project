# SATNAC — Openserve EV Charging Site Decision Support (Finalist)

Finalist for the SATNAC Industry Solutions Challenge 2025.

## Project summary
A strategic decision‑support tool to guide Openserve’s transition to Electric Vehicles (EVs) in South Africa. The system evaluates potential charging locations (1,500+ candidates) using a hybrid geospatial + AI approach and produces a prioritized rollout segmentation (Green / Amber / Red) to help planners focus investments.

Key highlights:
- Evaluated 1,500+ candidate sites with a 7‑factor weighted scoring system.
- Factors: loadshedding risk, security, transformer capacity, operational cost, grid resilience, site footprint, connectivity.
- Used K‑Means clustering on weighted scores to generate Green/Amber/Red rollout categories.
- Delivered as a Python ML pipeline with an interactive web dashboard (Leaflet.js for maps, Chart.js for visualizations) for geospatial exploration and explainability.
- Reached the SATNAC 2025 finale — demonstrated strong modelling and explainability despite methodological limitations.

## What the project does
- Ingests geospatial site candidates and auxiliary datasets (grid assets, security layers, connectivity, transformer specs, cost estimates).
- Computes normalized scores per site across 7 decision factors.
- Applies weighted aggregation and K‑Means clustering to group sites into rollout categories.
- Exposes results via an interactive map dashboard supporting exploration, filtering, and basic explainability charts.

## Technical stack
- Python (data processing, feature engineering, ML pipeline)
- Common Python libs: pandas, geopandas, scikit‑learn, rasterio (or similar)
- Web dashboard: Leaflet.js + Chart.js (frontend), lightweight Python server (Flask/FastAPI/Streamlit – adapt to repo)
- GIS data formats: GeoJSON / Shapefiles, CSVs for attributes
- Tested on Windows (development environment notes below)

## Why combine AI and GIS
- Geospatial context is essential for EV planning: proximity to transformers, grid topology, population/connectivity and security are spatial.
- Artificial Intelligence and Machine Learning (clustering / scoring) helps synthesize multi‑dimensional factors into actionable segments.
- The dashboard makes complex spatial decisions transparent and traceable to component factors.

## Known limitations and methodological flaws
Be aware of these caveats observed during development and evaluation:
- Weight subjectivity: factor weights were expert‑informed but not optimized via robust stakeholder calibration or automated tuning.
- K‑Means assumptions: K‑Means presumes spherical clusters and equal variance; may misclassify spatially irregular or multimodal distributions.
- Data quality & granularity: transformer capacities, real operational load and real‑time grid constraints may be coarse or out‑of‑date.
- Validation: limited ground‑truth validation and field testing — segmentation is indicative, not definitive.
- Operational constraints: site permitting, land ownership, environmental impact and commercial negotiations are not modelled.
- Explainability: aggregated scores can obscure tradeoffs unless users inspect component factor contributions.

Suggested improvements (next steps)
- Replace or complement K‑Means with density/graph‑based clustering (DBSCAN, HDBSCAN) or supervised ranking if labeled outcomes become available.
- Implement sensitivity analysis and automated weight calibration (e.g., pairwise comparisons, analytic hierarchy process).
- Integrate higher‑fidelity grid models and near‑real‑time telemetry where available.
- Add cost‑benefit and ROI simulation, permit and land‑use constraints, and route/load forecasting.
- Collect field validation data to refine model and produce a formal evaluation metric suite.

## How to run locally (Windows)
Note: adapt commands to exact scripts in this repo.

1. Clone the repo
   git clone <repo-url>
   cd SATNAC_Openserve_Prototype

2. Create Python venv and install dependencies
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

3. Run data pipeline (example)
   python scripts/run_pipeline.py --input data/inputs --output results/

4. Run dashboard (example)
   python app/server.py
   # or
   streamlit run app/dashboard.py

Open http://localhost:5000 (or the port printed by the server) to view the interactive map.

## How to contribute
- Fork the repo and open a pull request with a clear description of changes.
- Preferred contributions:
  - Add unit/integration tests for pipeline steps
  - Improve scoring calibration & add sensitivity tests
  - Replace K‑Means with alternative clustering methods and compare results
  - Improve frontend UX (legend, cluster explainers, export features)
  - Add Dockerfile / CI for reproducible runs
- Coding standards: keep Python code PEP8 compliant; include tests for new functionality.

## Data & privacy
- The project combines public and internal datasets; ensure sensitive or proprietary grid data is handled per Openserve policies before sharing.
- Remove or anonymize sensitive attributes prior to publishing.

## Contact / provenance
- Developed for the SATNAC Industry Solutions Challenge 2025 — finalist.
- Use issues to report bugs or request features. Include dataset descriptions for reproducibility requests.
