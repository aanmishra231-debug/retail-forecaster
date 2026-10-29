
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATE_NAMES = [
    "date", "order_date", "sales_date", "datetime", "timestamp"
]
SALES_NAMES = [
    "sales", "weekly_sales", "revenue", "total_sales",
    "amount", "quantity", "units_sold"
]


def find_column(columns, candidates):
    normalised = {
        "".join(c.lower().split()).replace("_", ""): c
        for c in columns
    }
    for name in candidates:
        key = name.replace("_", "")
        if key in normalised:
            return normalised[key]
    return None


def load_sales_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put your original CSV at this location."
        )

    df = pd.read_csv(path)

    if df.empty or len(df.columns) < 2:
        raise ValueError("The CSV must contain date and sales columns.")

    date_col = find_column(df.columns, DATE_NAMES)

    if date_col is None:
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                date_col = col
                break

    sales_col = find_column(df.columns, SALES_NAMES)

    if date_col is None:
        raise ValueError(
            f"Cannot identify the date column. Found: {list(df.columns)}"
        )

    if sales_col is None:
        for col in df.columns:
            if col == date_col:
                continue
            values = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            if values.notna().mean() >= 0.8:
                sales_col = col
                df[col] = values
                break

    if sales_col is None:
        raise ValueError(
            f"Cannot identify a numeric sales column. Found: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(
        df[sales_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=[date_col, sales_col]).copy()
    df = df.sort_values(date_col)

    if df.empty:
        raise ValueError("No valid date and sales records were found.")

    info = type("DataInfo", (), {
        "date_col": date_col,
        "sales_col": sales_col,
        "rows": len(df),
        "date_min": df[date_col].min(),
        "date_max": df[date_col].max(),
    })()

    return df, info


def make_features(series):
    dates = pd.DatetimeIndex(series.index)

    X = pd.DataFrame(index=dates)
    X["day_of_week"] = dates.dayofweek
    X["day_of_month"] = dates.day
    X["month"] = dates.month
    X["day_of_year"] = dates.dayofyear
    X["week_of_year"] = dates.isocalendar().week.to_numpy(dtype=int)
    X["is_weekend"] = (dates.dayofweek >= 5).astype(int)

    for lag in [1, 7, 14, 28]:
        X[f"lag_{lag}"] = series.shift(lag).to_numpy()

    for window in [7, 14, 28]:
        X[f"rolling_mean_{window}"] = (
            series.shift(1)
            .rolling(window, min_periods=2)
            .mean()
            .to_numpy()
        )

    return X


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / denominator * 100
        if denominator else np.nan
    )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE (%)": wape,
    }


def evaluate_and_forecast(daily, date_col, sales_col, horizon=30):
    data = daily[[date_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])

    data = (
        data.groupby(date_col)[sales_col]
        .sum()
        .sort_index()
        .to_frame()
        .asfreq("D")
    )

    missing_days = int(data[sales_col].isna().sum())
    data[sales_col] = data[sales_col].fillna(0)

    y = data[sales_col].astype(float)

    if len(y) < 21:
        raise ValueError(
            f"Only {len(y)} days of data. At least 21 are needed."
        )

    horizon = max(1, min(int(horizon), 365))
    test_size = max(7, min(len(y) // 5, 60, len(y) // 3))

    train_y = y.iloc[:-test_size]
    test_y = y.iloc[-test_size:]

    features = make_features(y)
    train_X = features.loc[train_y.index].copy()
    test_X = features.loc[test_y.index].copy()

    valid = train_X.notna().all(axis=1)
    train_X = train_X.loc[valid]
    train_target = train_y.loc[train_X.index]

    if len(train_X) < 10:
        raise ValueError(
            "Not enough usable history after creating lag features. "
            "Try a dataset with more daily observations."
        )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_X, train_target)

    # Walk-forward, one-day-ahead validation:
    # lag features use sales known before each prediction date.
    predictions = []
    history = train_y.copy()

    for date in test_y.index:
        partial = history.reindex(
            pd.date_range(history.index.min(), date, freq="D")
        )
        X = make_features(partial).iloc[[-1]]
        X = X.reindex(columns=train_X.columns).fillna(0)

        prediction = max(0.0, float(model.predict(X)[0]))
        predictions.append(prediction)

        # Actual sales become known before the next validation day.
        history.loc[date] = test_y.loc[date]

    baseline = []
    history = train_y.copy()

    for date in test_y.index:
        prior_date = date - pd.Timedelta(days=7)
        value = history.loc[prior_date] if prior_date in history.index else history.iloc[-1]
        baseline.append(max(0.0, float(value)))
        history.loc[date] = test_y.loc[date]

    metrics = pd.DataFrame([
        {
            "Model": "7-day seasonal baseline",
            **calculate_metrics(test_y, baseline),
        },
        {
            "Model": "Random Forest",
            **calculate_metrics(test_y, predictions),
        },
    ])

    validation = pd.DataFrame({
        "date": test_y.index,
        "actual_sales": test_y.to_numpy(),
        "seasonal_naive_prediction": baseline,
        "model_prediction": predictions,
    })

    # Fit on all available history for future forecasting.
    all_features = make_features(y)
    usable = all_features.notna().all(axis=1)
    final_X = all_features.loc[usable]
    final_y = y.loc[final_X.index]

    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(final_X, final_y)

    future_dates = pd.date_range(
        y.index.max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    combined = y.copy()
    future_values = []

    for date in future_dates:
        calendar = pd.date_range(
            combined.index.min(), date, freq="D"
        )
        extended = combined.reindex(calendar)
        X = make_features(extended).iloc[[-1]]
        X = X.reindex(columns=final_X.columns).fillna(0)

        prediction = max(0.0, float(final_model.predict(X)[0]))
        future_values.append(prediction)
        combined.loc[date] = prediction

    forecast = pd.DataFrame({
        "date": future_dates,
        "forecast_sales": future_values,
    })

    history_df = data.reset_index()
    history_df = history_df.rename(columns={"index": date_col})

    summary = {
        "observations": len(y),
        "missing_days_filled_with_zero": missing_days,
        "training_end": str(train_y.index.max().date()),
        "validation_start": str(test_y.index.min().date()),
        "validation_end": str(test_y.index.max().date()),
        "forecast_start": str(future_dates.min().date()),
        "forecast_end": str(future_dates.max().date()),
    }

    return history_df, validation, metrics, forecast, summary
