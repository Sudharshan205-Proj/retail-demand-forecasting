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

with st.form("forecast_form"):
    st.subheader("Store & product")
    col1, col2 = st.columns(2)

    with col1:
        family = st.selectbox("Product family", options=FAMILIES)
        store_type = st.selectbox("Store type", options=STORE_TYPES, index=3)
        cluster = st.number_input("Store cluster (1-17)", min_value=1, max_value=17, value=1, step=1)

    with col2:
        forecast_date = st.date_input("Forecast date", value=dt.date.today())
        transactions_lag_1 = st.number_input(
            "Store's transaction count yesterday", min_value=0.0, value=1500.0, step=50.0
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
        "transactions_lag_1": transactions_lag_1,
        f"family_{family}": 1,
        f"store_type_{store_type}": 1,
    }

    unsupplied = [
        c for c in feature_cols
        if c not in row and not c.startswith(("family_", "store_type_"))
    ]
    if unsupplied:
        st.error(
            "The saved model expects features this form doesn't collect: "
            f"{', '.join(unsupplied)}. Re-run `retail-demand-forecasting.ipynb` to regenerate "
            "`models/final_model.joblib` and `models/feature_cols.joblib`."
        )
        st.stop()

    input_df = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])

    forecast = max(0.0, float(model.predict(input_df)[0]))

    same_weekday_avg = (sales_lag_7 + sales_lag_14 + sales_lag_28) / 3
    if same_weekday_avg > 0:
        reference, reference_label = same_weekday_avg, "same-weekday average (last 3 weeks)"
    else:
        reference, reference_label = (rolling_mean_7 + rolling_mean_28) / 2, "recent average"

    st.divider()
    st.metric(
        "Forecasted units sold",
        f"{forecast:,.1f}",
        delta=f"{forecast - reference:+.1f} vs. {reference_label}",
    )

    if forecast > reference * 1.2:
        st.success(f"📈 Forecast is notably above the {reference_label} — consider extra stock.")
    elif forecast < reference * 0.8:
        st.warning(f"📉 Forecast is notably below the {reference_label} — consider reducing stock.")
    else:
        st.info(f"➡️ Forecast is in line with the {reference_label}.")

    with st.expander("Engineered features sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Demo only — trained on the Store Sales (Corporación Favorita) dataset through "
    "2017-08-15. Not validated against more recent or out-of-country retail data. "
    "See the report's Limitations section."
)
