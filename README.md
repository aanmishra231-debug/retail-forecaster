
# Retail Sales Forecasting

A Python project for retail sales analysis, machine-learning forecasts,
model evaluation, and visualization.

## Dataset

Place the original `sales_data.csv` in the repository root.
The project reads this file and does not overwrite it.

## Features

- Automatic date and sales column detection
- Daily sales aggregation
- Historical sales visualization
- Random Forest forecasting
- Seven-day seasonal baseline comparison
- Chronological validation
- MAE, RMSE, and WAPE evaluation
- Future forecasts and downloadable CSV results
- Streamlit dashboard

## Run in GitHub Codespaces

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app.py
```

Run forecasting from the terminal:

```bash
python train.py --data sales_data.csv --horizon 30
```

Results are saved in the `outputs/` directory.

## Important

Forecasts are estimates, not guarantees. Check the validation
metrics before relying on predictions for business decisions.
