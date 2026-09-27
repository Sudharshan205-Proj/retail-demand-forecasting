"""
Retail Demand Forecasting — Interactive Demo.

Run with:
    streamlit run streamlit_app.py
"""

import datetime as dt

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retail Demand Forecasting Demo",
                   page_icon="📦", layout="centered")

MODEL_PATH = "models/final_model.joblib"
FEATURE_COLS_PATH = "models/feature_cols.joblib"
STORES_PATH = "data/stores.csv"

FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
    "CELEBRATION", "CLEANING", "DAIRY", "DELI", "EGGS", "FROZEN FOODS",
    "GROCERY I", "GROCERY II", "HARDWARE", "HOME AND KITCHEN I", "HOME AND KITCHEN II",
    "HOME APPLIANCES", "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
    "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE", "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS", "PRODUCE",
    "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
]
STORE_TYPES = ["A", "B", "C", "D", "E"]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, feature_cols


@st.cache_data
def load_store_metadata():
    stores = pd.read_csv(STORES_PATH, dtype={"store_nbr": "int16", "cluster": "int16"})
    return stores.set_index("store_nbr")[["city", "state", "type", "cluster"]]


st.title("📦 Retail Demand Forecasting — Live Sales Forecast")
st.caption(
    "Pick a store and product family, enter recent sales history and promotion/holiday "
    "details, and get a live daily unit-sales forecast from the trained model "
    "(Store Sales / Corporación Favorita dataset)."
)

try:
    model, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `retail-demand-forecasting.ipynb` first — "
        "it saves `models/final_model.joblib` and `models/feature_cols.joblib` "
        "after the final-model-selection step."
    )
    st.stop()

try:
    store_metadata = load_store_metadata()
except FileNotFoundError:
    st.error(
        "`data/stores.csv` not found. Download the Store Sales dataset into `data/` "
        "(see the README's Setup section) — the demo looks up each store's cluster "
        "and type from this file, the same way the notebook does."
    )
    st.stop()

with st.form("forecast_form"):
    st.subheader("Store & product")
    col1, col2 = st.columns(2)

    with col1:
        store_nbr = st.selectbox("Store number", options=sorted(store_metadata.index.tolist()))
        family = st.selectbox("Product family", options=FAMILIES)

    with col2:
        forecast_date = st.date_input("Forecast date", value=dt.date.today())
        transactions = st.number_input(
            "Recent daily transaction count for this store", min_value=0.0, value=1500.0, step=50.0
        )

    st.subheader("Recent sales history for this store × family")
    st.caption(
        "These stand in for the lag/rolling features the notebook computes from a live "
        "per-series history — see the report's Limitations section."
    )
    col3, col4, col5 = st.columns(3)

    with col3:
        sales_lag_7 = st.number_input("Sales 7 days ago", min_value=0.0, value=10.0, step=1.0)
        sales_lag_14 = st.number_input("Sales 14 days ago", min_value=0.0, value=10.0, step=1.0)

    with col4:
        sales_lag_28 = st.number_input("Sales 28 days ago", min_value=0.0, value=10.0, step=1.0)
        rolling_mean_7 = st.number_input(
            "Average daily sales, last 7 days", min_value=0.0, value=10.0, step=1.0
        )

    with col5:
        rolling_mean_28 = st.number_input(
            "Average daily sales, last 28 days", min_value=0.0, value=10.0, step=1.0
        )
        rolling_std_7 = st.number_input(
            "Std. dev. of daily sales, last 7 days", min_value=0.0, value=3.0, step=0.5
        )

    st.subheader("Promotions & holidays")
    col6, col7 = st.columns(2)

    with col6:
        onpromotion = st.number_input(
            "Items on promotion that day", min_value=0, value=0, step=1
        )
    with col7:
        is_holiday = st.checkbox("This date is a national holiday / non-working day")

    submitted = st.form_submit_button("Forecast sales")

if submitted:
    # Re-derive exactly the same engineered features the notebook computes,
    # from the recent-history and calendar fields a user can plausibly supply
    # for a single store x family x date forecast.
    store_row = store_metadata.loc[store_nbr]
    cluster = int(store_row["cluster"])
    store_type = str(store_row["type"])

    dayofweek = forecast_date.weekday()
    month = forecast_date.month
    day = forecast_date.day
    is_weekend = int(dayofweek >= 5)
    any_promotion = int(onpromotion > 0)

    row = {
        "onpromotion": onpromotion,
        "any_promotion": any_promotion,
        "is_holiday": int(is_holiday),
        "cluster": cluster,
        "sales_lag_7": sales_lag_7,
        "sales_lag_14": sales_lag_14,
        "sales_lag_28": sales_lag_28,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_28": rolling_mean_28,
        "rolling_std_7": rolling_std_7,
        "dayofweek": dayofweek,
        "month": month,
        "day": day,
        "is_weekend": is_weekend,
        "transactions": transactions,
        f"family_{family}": 1,
        f"store_type_{store_type}": 1,
    }

    # Build the row in the exact column order the model was trained on,
    # filling anything the notebook's feature set has but this form doesn't
    # (every other family_* / store_type_* dummy column stays 0).
    input_df = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])

    forecast = float(model.predict(input_df)[0])
    recent_avg = (rolling_mean_7 + rolling_mean_28) / 2

    st.divider()
    st.metric(
        "Forecasted units sold",
        f"{forecast:,.1f}",
        delta=f"{forecast - recent_avg:+.1f} vs. recent average",
    )

    if forecast > recent_avg * 1.2:
        st.success("📈 Forecast is notably above recent average — consider extra stock.")
    elif forecast < recent_avg * 0.8:
        st.warning("📉 Forecast is notably below recent average — consider reducing stock.")
    else:
        st.info("➡️ Forecast is in line with recent average sales.")

    with st.expander("Engineered features sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Demo only — trained on the Store Sales (Corporación Favorita) dataset through "
    "2017-08-15. Not validated against more recent or out-of-country retail data. "
    "See the report's Limitations section."
)
