# VNL Tournament Simulator 🏐

An AI-powered volleyball tournament simulator built on the VNL (Volleyball Nations League) dataset.

> ⚠️ This project is currently in active development. The notebook contains working prototypes for all core components. The Streamlit UI and final deployment are in progress.

---

## Current Progress

### Data exploration and aggregation
- Loaded and explored the VNL dataset (305 players, 18 teams, 19 statistical columns)
- Aggregated individual player statistics per team using sums
- Derived three efficiency metrics: Attack Efficiency, Serve Efficiency, Block Efficiency

### Team Strength Score
- Normalized all statistics between 0 and 1 using MinMaxScaler
- Weighted positive stats (points, efficiency) vs negative stats (errors)
- Generated a strength ranking for all 18 teams

### Match and tournament simulation
- Implemented logistic-function-based win probability per match
- Simulated set-by-set scores (best of 5 format)
- Implemented full group stage (round-robin, 4 groups of 4)
- Implemented knockout stage (quarterfinals, semifinals, third place, final)

### SHAP explainability
- Generated synthetic match data by simulating team pair matchups
- Trained an XGBoost classifier on stat differences between teams
- Used SHAP TreeExplainer to identify top 3 influential statistics per match

### HuggingFace natural language analysis
- Integrated SmolLM2-1.7B-Instruct via AutoModelForCausalLM
- Used chat template format for structured prompting
- Generates a 3-4 sentence match analysis based on result and SHAP values

---

## Dataset

**Source:** [VNL 2025 Player Data — Kaggle](https://www.kaggle.com/datasets/joshuali12/vnl-2025-player-data)

305 players across 16 national teams with 19 statistical columns covering attack, block, serve, set, dig, and receive performance.

---

## Tech Stack so far

| Component | Technology |
|---|---|
| Data processing | pandas |
| Normalization | scikit-learn (MinMaxScaler) |
| Prediction model | XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Natural language generation | HuggingFace Transformers (SmolLM2-1.7B-Instruct) |

---

## Project Structure

```
vnl-tournament-simulator/
├── data/
│   └── dataset.csv              # VNL player statistics
├── notebooks/
│   └── exploration.ipynb        # All prototyping done here so far
├── src/                         # To be populated from notebook code
│   ├── predictor.py
│   ├── tournament.py
│   ├── shap_explainer.py
│   └── nlp.py
├── app.py                       # Streamlit UI — in progress
└── requirements.txt
```

---

## Author

Alexei Luchian