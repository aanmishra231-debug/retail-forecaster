
import argparse
from pathlib import Path

from src.forecasting import load_sales_csv, evaluate_and_forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sales_data.csv")
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df, info = load_sales_csv(args.data)

    print(f"Loaded {info.rows} rows")
    print(f"Date column: {info.date_col}")
    print(f"Sales column: {info.sales_col}")
    print(f"Date range: {info.date_min} to {info.date_max}")

    daily = (
        df.groupby(info.date_col, as_index=False)[info.sales_col]
        .sum()
        .sort_values(info.date_col)
    )

    history, validation, metrics, forecast, summary = (
        evaluate_and_forecast(
            daily,
            info.date_col,
            info.sales_col,
            args.horizon,
        )
    )

    history.to_csv(output_dir / "daily_sales.csv", index=False)
    validation.to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    forecast.to_csv(output_dir / "future_forecast.csv", index=False)

    with open(output_dir / "run_summary.txt", "w") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    print("\nModel evaluation:")
    print(metrics.round(3).to_string(index=False))
    print(f"\nForecast saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
