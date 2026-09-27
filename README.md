# 📦 Retail Demand Forecasting

**Data Science Internship Project — Codec Technologies**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-006400)
![Prophet](https://img.shields.io/badge/Prophet-1.1-0072B2)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

A demand-forecasting pipeline built on [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting), a ~3-million-row multi-series retail dataset from Corporación Favorita, a 54-store Ecuadorian grocery chain. Nine modeling experiments — naive baselines, per-series classical models (ARIMA, Prophet), and global machine-learning/deep-learning models trained across all ~1,782 store × product-family series at once — are compared head-to-head on the same held-out 15-day window, and the winner is picked fairly rather than assumed.

**Headline result:** the final model forecasts daily unit sales across all 1,782 store × family series with a **WAPE of 12.20%** and **RMSE of 189.65** — well inside the project's own success criteria — versus a naive-baseline WAPE of 15.77%.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Internship & Course Background](#internship--course-background)
- [The Six-Phase Analysis Framework](#the-six-phase-analysis-framework)
- [Project Roadmap & Timeline](#project-roadmap--timeline)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Modeling Techniques, Tools & Metrics](#modeling-techniques-tools--metrics)
- [Tools & Tech Stack by Phase](#tools--tech-stack-by-phase)
- [Visualization Reference](#visualization-reference)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results at a Glance](#results-at-a-glance)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Dataset & Citation](#dataset--citation)

---

## About the Project

> ### Retail Demand Forecasting
> **Goal:** Forecast product demand to optimize inventory, using historical sales data, promotions, and holidays as inputs.
>
> **Guidelines:**
> - Use ARIMA, Prophet, or LSTM for time-series prediction
> - Evaluate with RMSE or MAPE

That is the original brief for this internship project. Everything else in this repository — the 9-experiment comparison matrix spanning classical, global-ML and deep-learning tracks, the leakage-safe lag/rolling feature engineering, the fair WAPE-optimized final-model selection among deployable global models, and the interactive demo — is this brief carried out end-to-end on a real ~3-million-row, 1,782-series dataset, following the same six-phase analysis cycle used in the internship's [Fraud Detection System](../fraud-detection-system) project and its foundational coursework (see below).

|---|---|
| **Task** | Multi-series regression — forecast daily unit sales per store × product-family combination, 15 days ahead |
| **Dataset** | Store Sales (Corporación Favorita), 3,000,888 training rows, 1,782 store × family series across 54 stores and 33 product families |
| **Approach** | 9-experiment comparison matrix (naive, ARIMA, Prophet, Random Forest, XGBoost ×2, tuned XGBoost, LSTM) + fair WAPE-optimized selection among deployable global models |
| **Final model** | XGBoost (global, untuned) — `n_estimators=400, max_depth=8, learning_rate=0.05` |
| **Result** | RMSE = 189.65 · MAPE = 42.43% · **WAPE = 12.20%** · Bias = +2.22% |
| **Full write-up** | [`retail_demand_forecasting_report.md`](./retail_demand_forecasting_report.md) |

---

## Internship & Course Background

This project is a **capstone-style follow-on** to the 8-week Data Science internship at **Codec Technologies**, reusing the same six-phase workflow and toolchain established in the internship's [Fraud Detection System](../fraud-detection-system) project.

### Internship Curriculum

| Week | Focus | Ties to This Project |
|---|---|---|
| 1 | Introduction to Data Science | Framed the business task and reused the six-phase analysis cycle from the fraud project |
| 2 | Data Cleaning and Preprocessing | Oil-price interpolation, transferred-holiday resolution, series-completeness checks |
| 3 | Exploratory Data Science (EDA) | Sales distribution, seasonality, promotion/holiday effects, correlation snapshot (Section 7 of the report) |
| 4 | Advanced Data Science | Lag/rolling feature engineering computed per series, chronological train/val/test methodology |
| 5 | Machine Learning Basics | Naive/seasonal-naive baselines, ARIMA, Prophet |
| 6 | Applied Machine Learning | Global Random Forest & XGBoost experiment matrix, rolling-origin hyperparameter tuning |
| 7 | Big Data and Cloud Computing | Handling a 3M-row, 1,782-series dataset efficiently (dtype optimization, per-series groupby feature engineering) |
| 8 | **Final Project** | **This repository** — end-to-end multi-series demand-forecasting system |

### Foundational Coursework: Google Data Analytics Professional Certificate

| Week | Course | Core Content | Applied in This Project |
|---|---|---|---|
| 1 | Foundations: Data, Data, Everywhere | The 6-phase analysis process, analytical thinking, the data ecosystem | Structures the whole project end-to-end |
| 1 | Ask Questions to Make Data-Driven Decisions | SMART questions, problem framing, "prediction" as a problem archetype | Ask phase — business task & success criteria |
| 2 | Prepare Data for Exploration | The ROCCC data-quality framework, sampling/observer/confirmation bias, data ethics | Dataset ROCCC assessment, sales-vs-demand caveat |
| 3 | Process Data from Dirty to Clean | Data integrity, the "dirty data" taxonomy, cleaning workflows | Oil-price interpolation, transferred-holiday logic, leakage-safe lag features |
| 4 | Analyze Data to Answer Questions | Aggregation, GROUP BY-style analysis, pattern-finding, joins | EDA, sales-by-family/city/cluster breakdowns, the 9-experiment comparison matrix |
| 5 | Share Data Through the Art of Visualization | Chart selection, dashboarding, storytelling with data | Model-comparison bar chart, actual-vs-predicted plots, feature-importance plot, the Streamlit demo |
| 6 | Data Analysis with R Programming | Tidyverse/dplyr, ggplot2, R Markdown, reproducibility | Reproducible environment practices (`requirements.txt`, notebook) — implemented in **Python** rather than R |
| 7 | Google Data Analytics Capstone | End-to-end case study structure, portfolio packaging | This report's problem → data → methods → results → recommendations |

The certificate teaches this six-phase workflow using spreadsheets, SQL, Tableau, and R.
**This project implements the same six-phase cycle end-to-end in Python** — pandas/NumPy in place of spreadsheets, statsmodels/Prophet/scikit-learn/XGBoost/TensorFlow in place of a BI tool, and Jupyter/GitHub in place of Tableau dashboards.

---

## The Six-Phase Analysis Framework

Every phase of this project maps directly onto the cycle used throughout the Google Data Analytics Certificate and the fraud-detection project before it:

```mermaid
%%{init: {"flowchart": {"curve": "basis"}, "themeVariables": {"fontSize": "16px"}}}%%
flowchart LR
    A(["🎯<br/><b>ASK</b><br/>Define the<br/>business task"])
    B(["📥<br/><b>PREPARE</b><br/>Source & assess<br/>the data"])
    C(["🧹<br/><b>PROCESS</b><br/>Clean & engineer<br/>features"])
    D(["🔬<br/><b>ANALYZE</b><br/>Model &<br/>evaluate"])
    E(["📊<br/><b>SHARE</b><br/>Visualize &<br/>communicate"])
    F(["✅<br/><b>ACT</b><br/>Recommend &<br/>conclude"])

    A ==> B ==> C ==> D ==> E ==> F
    F -. "🔁 iterate & refine" .-> A

    classDef ask     fill:#4C72B0,color:#ffffff,stroke:#2c3e5c,stroke-width:2px,rx:12,ry:12
    classDef prepare fill:#55A868,color:#ffffff,stroke:#2f5c3b,stroke-width:2px,rx:12,ry:12
    classDef process fill:#C44E52,color:#ffffff,stroke:#7a2f31,stroke-width:2px,rx:12,ry:12
    classDef analyze fill:#8172B2,color:#ffffff,stroke:#4d4370,stroke-width:2px,rx:12,ry:12
    classDef share   fill:#CCB974,color:#1a1a1a,stroke:#8f7c3f,stroke-width:2px,rx:12,ry:12
    classDef act     fill:#64B5CD,color:#1a1a1a,stroke:#376c7d,stroke-width:2px,rx:12,ry:12

    class A ask
    class B prepare
    class C process
    class D analyze
    class E share
    class F act
    linkStyle 5 stroke:#999,stroke-width:1.5px,stroke-dasharray:4 3
```

| Phase | What It Means Here | Primary Output |
|---|---|---|
| **Ask** | Define the demand-forecasting business task and SMART success criteria (WAPE ≤ 15%, RMSE below naive baseline for ≥80% of families) | Problem statement (Report) |
| **Prepare** | Download the 6-file Kaggle dataset, assess it against ROCCC, document schema, series completeness, and the sales-vs-demand distinction | Dataset overview (Report) |
| **Process** | Oil-price interpolation, transferred-holiday resolution, per-series lag/rolling feature engineering, chronological train/val/test split | Modeling-ready feature set (Report) |
| **Analyze** | Run the 9-experiment matrix (naive → classical → global ML → deep learning), tune the strongest global candidate, select fairly among deployable models | Results table + selected model (Report) |
| **Share** | Model-comparison bar chart, actual-vs-predicted plots, correlation heatmap, feature-importance plot, optional Streamlit demo | 18 result visualizations (`assets/`) |
| **Act** | Recommend inventory adjustments by forecast bias, document limitations, package as a portfolio-ready GitHub repository | Recommendations & next steps (Report) |

---

## Project Roadmap & Timeline

### Suggested Timeline (from the project plan)

| Week | Phase(s) | Planned Focus |
|---|---|---|
| Week 1 | Ask + Prepare | Problem framing, dataset download/assessment, environment setup, initial EDA across all 6 files |
| Week 2 | Process | Date/holiday/promotion cleaning, oil-price interpolation, lag/rolling/calendar feature engineering, chronological split |
| Week 3 | Analyze | Naive baseline, ARIMA and Prophet on representative series, first global tree-based models |
| Week 4 | Analyze (cont.) | LSTM track, hyperparameter tuning via rolling-origin cross-validation, final model selection |
| Week 5 | Share | Visualizations, optional Streamlit demo, written report and GitHub packaging |
| Week 6 | Act + Presentation | Recommendations, limitations, internship presentation/demo, polish |

### Actual Execution Roadmap

In practice, the classical (ARIMA/Prophet) and global-ML tracks were developed together in a single notebook, and the report/README went through a reconciliation pass once the real results came in — including the finding that hyperparameter tuning did not actually improve on the untuned model. Each stage below is color-coded to match its phase in the six-phase framework:

```mermaid
%%{init: {"flowchart": {"curve": "basis"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart TD
    subgraph W1["🎯📥&nbsp; WEEK 1 &nbsp;·&nbsp; Ask + Prepare"]
        direction TB
        A1(["Define business task<br/>& SMART criteria"])
        A2(["Set up venv, Git<br/>& Kaggle access"])
        A3(["Download 6-file dataset<br/>~3M rows, 1,782 series"])
        A4(["Dataset overview &<br/>series-completeness checks"])
        A1 --> A2 --> A3 --> A4
    end

    subgraph W2["🧹&nbsp; WEEK 2 &nbsp;·&nbsp; Process"]
        direction TB
        B1(["Resolve transferred-<br/>holiday flag"])
        B2(["Interpolate oil-price<br/>gaps (31.1% missing)"])
        B3(["Engineer per-series lag,<br/>rolling & calendar features"])
        B4(["Chronological split<br/>— never a random shuffle"])
        B1 --> B2 --> B3 --> B4
    end

    subgraph W34["🔬&nbsp; WEEKS 3–4 &nbsp;·&nbsp; Analyze"]
        direction TB
        C1(["Naive & seasonal-naive<br/>baseline"])
        C2(["Classical track<br/>ARIMA + Prophet, 5 series"])
        C3(["Global ML track: Random Forest<br/>& XGBoost (± oil)"])
        C4(["LSTM track<br/>store/family embeddings"])
        C5(["Rolling-origin tuning +<br/>fair final selection"])
        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph W5["📊&nbsp; WEEK 5 &nbsp;·&nbsp; Share"]
        direction TB
        D1(["Model-comparison bar chart<br/>& actual-vs-predicted plots"])
        D2(["Feature-importance<br/>plot"])
        D3(["Streamlit demo"])
        D4(["Written report<br/>+ README"])
        D1 --> D2 --> D3 --> D4
    end

    subgraph W6["✅&nbsp; WEEK 6 &nbsp;·&nbsp; Act"]
        direction TB
        E1(["Recommendations<br/>& limitations"])
        E2(["Reconciliation pass:<br/>numbers, artifacts, .gitignore"])
        E3(["Final polish &<br/>GitHub packaging"])
        E1 --> E2 --> E3
    end

    W1 ==> W2
    W2 ==> W34
    W34 ==> W5
    W5 ==> W6

    classDef askprepare fill:#4C72B0,color:#ffffff,stroke:#2c3e5c,stroke-width:1.5px,rx:8,ry:8
    classDef process    fill:#C44E52,color:#ffffff,stroke:#7a2f31,stroke-width:1.5px,rx:8,ry:8
    classDef analyze    fill:#8172B2,color:#ffffff,stroke:#4d4370,stroke-width:1.5px,rx:8,ry:8
    classDef share      fill:#CCB974,color:#1a1a1a,stroke:#8f7c3f,stroke-width:1.5px,rx:8,ry:8
    classDef act        fill:#64B5CD,color:#1a1a1a,stroke:#376c7d,stroke-width:1.5px,rx:8,ry:8

    class A1,A2,A3,A4 askprepare
    class B1,B2,B3,B4 process
    class C1,C2,C3,C4,C5 analyze
    class D1,D2,D3,D4 share
    class E1,E2,E3 act

    style W1  fill:#eef2fa,stroke:#4C72B0,stroke-width:2px, color:#000000
    style W2  fill:#fceeee,stroke:#C44E52,stroke-width:2px, color:#000000
    style W34 fill:#f2eef8,stroke:#8172B2,stroke-width:2px, color:#000000
    style W5  fill:#fbf8ea,stroke:#CCB974,stroke-width:2px, color:#000000
    style W6  fill:#eaf5f9,stroke:#64B5CD,stroke-width:2px, color:#000000
```

---

## Machine Learning Workflow

The modeling pipeline itself — the heart of the Analyze phase — follows a leakage-aware, chronologically-split comparison workflow across three model tracks. Node shapes carry meaning throughout: **cylinders** are data at rest, **diamonds** are decisions/branch points, and the **pill-shaped node** is the final outcome.

```mermaid
%%{init: {"flowchart": {"curve": "basis"}, "themeVariables": {"fontSize": "14px"}}}%%
flowchart TD
    R[("Raw 6-file dataset<br/>3,000,888 sales rows")]
    S["Merge stores, oil, holidays,<br/>transactions onto sales"]
    FE["Feature engineering<br/>lags · rolling stats · calendar · promo · holiday"]
    SPLIT[["Chronological split<br/>Train to 2017-07-16 / Val 15d / Test 15d"]]

    R --> S --> FE --> SPLIT

    subgraph CLASSICAL["🔵 Classical Track — 5 representative series only"]
        U1["ARIMA (SARIMAX)<br/>weekly seasonal"]
        U2["Prophet<br/>+ holiday/promo regressors"]
    end

    subgraph GLOBAL["🟠 Global Track — all 1,782 series, one model"]
        V1["Naive / seasonal-naive<br/>baseline"]
        V2["Random Forest"]
        V3["XGBoost<br/>(± oil price)"]
        V4["LSTM<br/>store/family embeddings, 50-series sample"]
    end

    SPLIT ==> U1
    SPLIT ==> U2
    SPLIT ==> V1
    SPLIT ==> V2
    SPLIT ==> V3
    SPLIT ==> V4

    TUNE["RandomizedSearchCV<br/>rolling-origin CV on XGBoost"]
    V3 --> TUNE

    EVAL{{"Evaluate ALL 9 candidates<br/>on the SAME 15-day test window"}}
    U1 --> EVAL
    U2 --> EVAL
    V1 --> EVAL
    V2 --> EVAL
    V3 --> EVAL
    V4 --> EVAL
    TUNE --> EVAL

    DEPLOY{"Restrict to models with<br/>full-test-set predictions"}
    FINAL(["🏆 Final Model Selected<br/>XGBoost · global · untuned"])
    ART[("Persist artifacts<br/>model · feature_cols")]
    DEMO["Streamlit demo<br/>live sales forecast"]

    EVAL ==> DEPLOY ==> FINAL ==> ART ==> DEMO

    classDef data     fill:#e8eef7,stroke:#4C72B0,stroke-width:1.5px,color:#1a1a1a
    classDef decision fill:#fdf3d8,stroke:#b8973f,stroke-width:1.5px,color:#1a1a1a
    classDef process  fill:#f2eefa,stroke:#8172B2,stroke-width:1.5px,color:#1a1a1a
    classDef classical fill:#e3eefc,stroke:#3d6ea8,stroke-width:1.5px,color:#1a1a1a
    classDef global   fill:#fcecdd,stroke:#c47a2f,stroke-width:1.5px,color:#1a1a1a
    classDef final    fill:#2e9e6b,stroke:#1d5f41,stroke-width:2.5px,color:#ffffff

    class R,ART data
    class DEPLOY,EVAL decision
    class S,FE,SPLIT,TUNE,DEMO process
    class U1,U2 classical
    class V1,V2,V3,V4 global
    class FINAL final

    style CLASSICAL fill:#f4f8fd,stroke:#3d6ea8,stroke-width:2px
    style GLOBAL    fill:#fdf6ee,stroke:#c47a2f,stroke-width:2px
```

**Key design choices baked into this workflow:**
- The split is **chronological, never a random shuffle** — training only ever sees dates before validation/test, which is what makes lag/rolling features and the reported metrics trustworthy for a real forecasting deployment.
- ARIMA and Prophet satisfy the guideline's named classical techniques directly, fit on the **top 5 highest-volume series**; the global tree-based/LSTM tracks are the practical way to cover all ~1,782 series at once, matching how all four reference project videos approach this problem.
- The final model is chosen from **only the models with full-test-set predictions to actually deploy** — the representative-series-only classical/deep-learning experiments are compared for context but aren't candidates for the final artifact, since there's no global model behind them to save.

---

## Modeling Techniques, Tools & Metrics

### Experiment Matrix

| # | Model | Category | Scope | Library |
|---|---|---|---|---|
| 1a/1b | Naive / Seasonal-naive | Baseline heuristic | 5 representative series | — |
| 2 | ARIMA (SARIMAX) | Classical statistical | 5 representative series | `statsmodels` |
| 3 | Prophet | Additive decomposition + holiday/promo regressors | 5 representative series | `prophet` |
| 4 | Random Forest | Global machine learning (bagged trees) | All 1,782 series | `scikit-learn` |
| 5 | XGBoost | Global machine learning (boosted trees) | All 1,782 series | `xgboost` |
| 6 | XGBoost + oil price | Global machine learning | All 1,782 series | `xgboost` |
| 7 | LSTM | Deep learning, store/family embeddings | 50-series sample | `tensorflow` / `keras` |
| 8 | XGBoost (tuned) | Global machine learning + `RandomizedSearchCV` | All 1,782 series | `xgboost` + `scikit-learn` |

### Evaluation Metrics

| Metric | What It Measures | Why It Matters Here |
|---|---|---|
| **RMSE** | √(mean squared error) | Named in the guidelines; targets the mean, sensitive to large promotion-day spikes |
| **MAPE** | Mean absolute percentage error | Named in the guidelines, but explodes on the many zero-sales days common to slow-moving families |
| **WAPE** ⭐ | Weighted absolute percentage error (Σ\|error\| / Σ\|actual\|) | Primary guideline-adjacent metric — aggregates cleanly across all 1,782 series into one interpretable number, without MAPE's zero-division problem |
| **Forecast Bias** | (Σpredicted − Σactual) / Σactual | Directly answers "are we systematically over- or under-stocking?" — the business-facing tie-breaker |

MAPE is reported for every experiment (per the guideline) but was **not** used to pick the winner — Section 13 of the report explains why it's structurally misleading on this dataset's many zero-sales days, following the project-plan reference video's critique.

### Model Explainability

| Tool | Role |
|---|---|
| `.feature_importances_` (XGBoost) | Bar chart of which lag/calendar/exogenous features drive the winning model's forecasts |

---

## Tools & Tech Stack by Phase

| Phase | Tasks | Tools / Libraries |
|---|---|---|
| **Ask** | Problem framing, SMART success criteria | Markdown documentation (no code) |
| **Prepare** | Dataset download, schema/ROCCC assessment, dataset-overview functions | `pandas`, `numpy`, `kaggle` CLI |
| **Process** | Oil-price interpolation, holiday resolution, per-series lag/rolling/calendar feature engineering, chronological split | `pandas`, `numpy` |
| **Analyze** | Naive + classical + global ML + deep-learning tracks, hyperparameter tuning | `statsmodels`, `prophet`, `scikit-learn`, `xgboost`, `tensorflow`/`keras` |
| **Share** | Visualization, explainability, interactive demo | `matplotlib`, `seaborn`, `plotly`, `streamlit` |
| **Act** | Artifact persistence, reporting, packaging | `joblib`, Markdown, Git/GitHub |

### Full Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **Environment** | `venv` + `pip`, VS Code (Python + Jupyter extensions) |
| **Data handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Classical time series** | `statsmodels` (SARIMAX), `prophet` |
| **Machine learning** | `scikit-learn`, `xgboost` |
| **Deep learning** | `tensorflow` / `keras` (LSTM with embeddings) |
| **Deployment demo** | `streamlit` |
| **Data source** | `kaggle` CLI + API token |
| **Persistence** | `joblib` |
| **Version control** | Git + GitHub |

Exact pinned versions are in [`requirements.txt`](./requirements.txt).

---

## Visualization Reference

All 18 figures live in [`assets/`](./assets) and are generated directly by the notebook (each plotting cell saves to `assets/` before displaying inline).

| # | File | Plot Type | What It Shows |
|---|---|---|---|
| 01 | `01_total_sales_over_time.png` | Line chart | Total daily unit sales across all stores and families, 2013–2017 |
| 02 | `02_sales_distribution.png` | Histogram (log-scaled x-axis) | Distribution of nonzero daily sales |
| 03 | `03_sales_by_family.png` | Horizontal bar chart | Total sales by product family, top 15 |
| 04 | `04_sales_by_store_type.png` | Bar chart | Total sales by store type (A–E) |
| 05 | `05_oil_price_over_time.png` | Line chart | WTI oil price over time (gaps interpolated for plotting) |
| 06 | `06_promotion_effect.png` | Bar chart | Average sales, on promotion vs. not |
| 07 | `07_holiday_counts.png` | Bar chart | Holiday/event rows by type |
| 08 | `08_dayofweek_seasonality.png` | Line/bar chart | Average sales by day of week |
| 09 | `09_monthly_seasonality.png` | Line chart | Average sales by month |
| 10 | `10_sales_by_city.png` | Horizontal bar chart | Total sales by city, top 10 |
| 11 | `11_sales_by_cluster.png` | Bar chart | Total sales by store cluster |
| 12 | `12_holiday_effect.png` | Bar chart | Average sales, holiday/event date vs. regular day |
| 13 | `13_transactions_vs_sales_scatter.png` | Scatter plot (sampled) | Daily transactions vs. daily sales, per store |
| 14 | `14_raw_correlation_heatmap.png` | Heatmap | Correlation among raw daily aggregates, before feature engineering |
| 15 | `15_missing_days_distribution.png` | Histogram | Distribution of missing calendar days per store × family series |
| 16 | `16_model_comparison_bar.png` | Horizontal bar chart | WAPE across all experiments on the representative-series comparison |
| 17 | `17_actual_vs_predicted.png` | Line chart | Actual vs. predicted sales for a representative series over the test window |
| 18 | `18_feature_importance.png` | Horizontal bar chart | Top 15 features by importance for the final XGBoost model |

**Why these plot types:** line charts for anything sequential (total sales, oil price, monthly seasonality, actual-vs-predicted), bar/horizontal-bar charts for categorical comparisons (family, city, cluster, store type, model comparison), histograms for distributional questions (sales, missing days), a scatter plot for the two-variable transactions/sales relationship, a heatmap for many-variable correlation at a glance, and the feature-importance bar chart as the most stakeholder-friendly way to explain *what* drives the final model's forecasts.

---

## Repository Structure

```
retail-demand-forecasting/
├── retail-demand-forecasting.ipynb    # Full analysis: EDA → features → modeling → evaluation
├── exports/                           # Rendered exports of the notebook (HTML/PDF) — viewable without Jupyter
├── retail_demand_forecasting_report.md # In-depth written report (problem → data → methods → results → recs)
├── README.md                          # You are here
├── streamlit_app.py                   # Optional interactive demo — live sales forecast
├── requirements.txt                   # Pinned Python dependencies
├── project-hierarchy.txt              # Snapshot of the local project layout
├── .gitignore
├── assets/                            # 18 result figures used by the report/README (tracked)
├── data/                              # Store Sales CSVs — gitignored, see Setup
└── models/                            # Saved model artifacts — gitignored, generated by the notebook
```

## Setup

1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the dataset** — download the Store Sales dataset via the Kaggle competition CLI (requires a `kaggle.json` API token in `.kaggle/` or `~/.kaggle/`):
   ```bash
   kaggle competitions download -c store-sales-time-series-forecasting
   ```
   Unzip the 6 CSVs (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`) into `data/`.

## Usage

**Run the full analysis**
Open `retail-demand-forecasting.ipynb`, update `DATA_DIR` if your folder differs, and run all cells. This performs EDA, engineers leakage-safe lag/rolling/calendar features, runs all 9 experiments plus tuning, and saves the winning model to `models/`. Every plot is written to `assets/` automatically as it's generated.

Prefer not to install Jupyter first? Open the rendered export in `exports/` in any browser to see the fully executed notebook, outputs and all.

**Try the interactive demo**
Once the notebook has run at least once (so `models/final_model.joblib` exists) and `data/stores.csv` is present:
```bash
streamlit run streamlit_app.py
```
Pick a store and product family, enter recent sales history and promotion/holiday details, and get a live 1-day-ahead demand forecast. Note that lag and rolling-window features are entered directly rather than looked up live — see the report's Limitations section.

## Results at a Glance

| Experiment | RMSE | MAPE | WAPE | Bias |
|---|---|---|---|---|
| **5. XGBoost (global, untuned)** ⭐ | 189.65 | 42.43% | **12.20%** | +2.22% |
| 6. XGBoost + oil price | 202.29 | 41.53% | 13.14% | +3.78% |
| 8. XGBoost (tuned, global) | 206.37 | 62.04% | 13.76% | +4.10% |
| 4. Random Forest (global) | 229.12 | 42.91% | 14.83% | +4.08% |
| 1a. Naive (yesterday's value) | 2,063.52 | 16.22% | 15.77% | +1.93% |
| 3. Prophet (holiday + promo regressors) | 2,031.17 | 17.07% | 16.49% | +9.56% |
| 1b. Seasonal-naive (same weekday last week) | 2,503.25 | 21.75% | 21.14% | +8.66% |
| 2. ARIMA (SARIMAX, weekly seasonal) | 3,227.97 | 27.82% | 25.92% | +12.26% |
| 7. LSTM (50-series sample) | 500.63 | 74.75% | 27.01% | −10.61% |

The untuned global XGBoost model tops **both** the representative-series comparison above and the full-test-set comparison among deployable global models — no tuning tradeoff was needed here (see the report for why tuning actually underperformed).

<p align="center">
  <img src="assets/16_model_comparison_bar.png" alt="Model comparison by WAPE" width="500">
</p>

## Key Findings

- **Every one of the 1,782 store × family series has at least one missing calendar day** (mean 4 of 1,688 expected days) — gaps were left as-is rather than assumed to be zero-sales days.
- **31.1% of the full date range is missing an oil price** — interpolated/forward-filled before use, since Ecuador's oil-dependent economy makes this a meaningful (if weak, at the series level) exogenous signal.
- **Global tree-based models beat every per-series classical model by an order of magnitude on RMSE** (~190–230 vs. 2,000–3,200) — because their features (lag_7/14/28, rolling means, calendar, promotion, holiday) capture cross-series structure that a single-series ARIMA/Prophet fit cannot.
- **Hyperparameter tuning did not improve the final model** — the tuned XGBoost variant scored a *worse* WAPE (13.76%) than the untuned one (12.20%), because `RandomizedSearchCV` optimized for RMSE on rolling-origin CV folds, which didn't track WAPE on this specific 15-day holdout. Reported as a genuine finding rather than hidden.
- **Oil price as a feature slightly hurt XGBoost's WAPE** (13.14% vs. 12.20% without it) — consistent with the project plan's expectation that oil is a weak, indirect signal at the individual store × family level.

## Limitations

Sales here are realized sales, not true demand (stockouts can under-report demand); only National-level holidays were applied uniformly, leaving out Regional/Local holidays; ARIMA, Prophet, and the LSTM were validated only on a sample of series (5 and 50, respectively) rather than the full 1,782; and the dataset ends in 2017 and reflects Ecuador-specific seasonality and an oil-dependent economy. Full discussion, including why MAPE is reported but not used to select the winner, is in the report.

## Dataset & Citation

Corporación Favorita. "Store Sales - Time Series Forecasting." *Kaggle*, 2021. Available on [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).

## Streamlit Deployment

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://internship-retail-demand-forecasting.streamlit.app/)

## Author

**Sudharshan Moodley**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sudharshan_Moodley-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sudharshan-moodley-0a1a5b2a9/)
[![GitHub](https://img.shields.io/badge/GitHub-Sudharshan205--Proj-181717?logo=github&logoColor=white)](https://github.com/Sudharshan205-Proj)