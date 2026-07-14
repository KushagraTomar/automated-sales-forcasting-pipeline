# Automated Sales Forecasting Pipeline

This project implements the idea discussed in the shared Claude conversation: an automated sales forecasting pipeline using XGBoost, MLflow, and Airflow.

## Goal

Build a retraining workflow that:
- loads sales data,
- engineers lag and rolling features,
- trains an XGBoost regressor,
- evaluates a challenger model against the current production baseline,
- promotes the challenger when it improves the holdout metric.

## Dataset

The starter design uses the Store Item Demand Forecasting Challenge dataset, which contains daily sales for 10 stores and 50 items over 5 years.

## Project structure

- `dags/` for Airflow orchestration
- `src/` for pipeline logic
- `config/` for configurable settings
- `tests/` for lightweight unit tests
