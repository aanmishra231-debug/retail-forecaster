
from pathlib import Path

import pandas as pd
import streamlit as st

from src.forecasting import load_sales_csv, evaluate_and_forecast

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("Retail Sales Forecasting")
st.caption("Sales analysis, model evaluation, and future demand predictions.")

with st.sidebar:
    st.header("Settings")
    csv_path = st.text_input("Dataset path", "sales_data.csv")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
    run = st.button("Run forecast", type="primary")

if not Path(csv_path).exists():
    st.error(
        f"Dataset not found: {csv_path}. "
        "Place your original sales_data.csv in the repository root."
    )
    st.stop()

try:
    raw, info = load_sales_csv(csv_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

a, b, c = st.columns(3)
a.metric("Rows loaded", f"{info.rows:,}")
b.metric("First date", str(info.date_min.date()))
c.metric("Last date", str(info.date_max.date()))

with st.expander("Inspect original data"):
    st.write("Date column:", info.date_col)
    st.write("Sales column:", info.sales_col)
    st.write("Available columns:", list(raw.columns))
    st.dataframe(raw.head(20), use_container_width=True)

if run:
    with st.spinner("Training and evaluating forecasting models..."):
        try:
            daily = (
                raw.groupby(info.date_col, as_index=False)[info.sales_col]
                .sum()
                .sort_values(info.date_col)
            )

            history, validation, metrics, forecast, summary = (
                evaluate_and_forecast(
                    daily,
                    info.date_col,
                    info.sales_col,
                    horizon,
                )
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.session_state["forecast_results"] = (
        history, validation, metrics, forecast, summary
    )

if "forecast_results" not in st.session_state:
    st.info("Click **Run forecast** to generate results.")
    st.stop()

history, validation, metrics, forecast, summary = (
    st.session_state["forecast_results"]
)

st.subheader("Historical sales")
history = history.copy()
history["date"] = pd.to_datetime(history[info.date_col])
st.line_chart(
    history.set_index("date")[info.sales_col].rename("Sales")
)

st.subheader("Model accuracy")
st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
st.caption(
    "Validation uses a later time period than the training period. "
    "The metrics help compare forecasts; lower MAE and RMSE are better."
)

st.subheader("Actual vs predicted sales")
validation_chart = validation.set_index("date")[
    ["actual_sales", "seasonal_naive_prediction", "model_prediction"]
]
st.line_chart(validation_chart)

st.subheader(f"Forecast for the next {len(forecast)} days")
st.line_chart(forecast.set_index("date")["forecast_sales"])

m1, m2 = st.columns(2)
m1.metric("Forecast total", f"{forecast['forecast_sales'].sum():,.2f}")
m2.metric("Forecast daily average", f"{forecast['forecast_sales'].mean():,.2f}")

st.download_button(
    "Download forecast CSV",
    forecast.to_csv(index=False).encode("utf-8"),
    file_name="future_forecast.csv",
    mime="text/csv",
)

st.download_button(
    "Download evaluation metrics",
    metrics.to_csv(index=False).encode("utf-8"),
    file_name="model_metrics.csv",
    mime="text/csv",
)

with st.expander("Forecast details"):
    st.json(summary)
