import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pymysql
import shap
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from dagster import (
    Array,
    Definitions,
    Field,
    Int,
    OpExecutionContext,
    String,
    asset,
    job,
    op,
)

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@asset(
    config_schema={
        "host": Field(String, description="External DB host (e.g., Biggs' hostname)"),
        "port": Field(Int, description="External DB port (usually 3306 for MariaDB)"),
        "user": Field(String, description="Username for the external database"),
        "password": Field(String, description="Password for the external database"),
        "database": Field(String, description="Database name to connect to"),
        "queries": Field(Array(String), description="List of SQL queries to run"),
    }
)
def external_data(context) -> dict:
    config = context.op_config
    results = {}
    try:
        conn = pymysql.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )
        context.log.info("Connected to external MariaDB database.")
        for i, query in enumerate(config["queries"]):
            df = pd.read_sql(query, conn)
            results[f"query_{i}"] = df
            context.log.info(f"Query {i} executed; retrieved {len(df)} rows.")
    except Exception as e:
        context.log.error(f"Error fetching data: {e}")
        raise e
    finally:
        if conn:
            conn.close()
            context.log.info("Connection closed.")
    return results


@op
def print_external_data_head(context, ext_data: dict) -> dict:
    # Print head(5) of each dataframe in the external data dictionary
    for key, df in ext_data.items():
        context.log.info(f"Head of {key}:")
        context.log.info(df.head(5).to_string())
    # Return the data unchanged in case downstream ops need it
    return ext_data


@asset
def biggs_dataset() -> pd.DataFrame:
    # Generate one year of daily data
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    np.random.seed(42)
    # Create a time series target
    target = np.random.rand(365)
    df = pd.DataFrame({"target": target}, index=dates)
    # Create a lag feature (previous day's target)
    df["lag_1"] = df["target"].shift(1)
    # Drop the first row that has a NaN lag value
    df = df.dropna()
    return df


@asset
def split_data(biggs_dataset: pd.DataFrame):
    X = biggs_dataset.drop(columns=["target"])
    y = biggs_dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@op(
    config_schema={
        "algorithm": Field(String, default_value="xgboost", is_required=False)
    }
)
def train_model(context, split_data):
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]

    context.log.info(
        f"X_train columns: {X_train.columns.tolist()}, shape: {X_train.shape}"
    )

    algo = context.op_config["algorithm"]
    if algo == "xgboost":
        context.log.info("xgboost selected")
        model = XGBRegressor(
            n_estimators=100, random_state=42, objective="reg:squarederror"
        )
    elif algo == "lightgbm":
        context.log.info("lightgbm selected")
        model = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    context.log.info("Entering model.fit function.")
    model.fit(X_train, y_train)
    return (model, algo)


@op
def predict(context, model_tuple, split_data):
    model, algo = model_tuple  # Unpack the tuple to get the model instance.
    X_test = split_data["X_test"]
    y_test_pred = model.predict(X_test)
    return {"y_test_pred": y_test_pred}


@op
def log_to_mlflow(context: OpExecutionContext, train_model, split_data, predict):
    logger.add("mlflow_training.log", rotation="1 MB", level="INFO")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    # Now predict returns test set predictions for regression
    y_test_pred = predict["y_test_pred"]

    (model, algo) = train_model
    model_name = "biggs_" + algo + "_model"

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run started: {run_id}")

        # Log the model and parameters
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_params({"n_estimators": 100, "random_state": 42})

        # Compute R^2 scores as a metric for regression
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)

        # Generate a SHAP summary plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("shap_summary.png", bbox_inches="tight")
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        # Generate an Evidently report for regression
        report_data = pd.DataFrame({"target": y_test, "prediction": y_test_pred})
        report = Report(metrics=[RegressionPreset()])
        report.run(reference_data=report_data, current_data=report_data)
        report.save_html("evidently_report.html")
        mlflow.log_artifact("evidently_report.html")

        # Register the model
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{model_name}",
            name=f"biggs_{algo}",
        )

        logger.info(f"Model registered: {result.name}, version: {result.version}")
        logger.info(f"Train Score: {train_score}, Test Score: {test_score}")


@op
def log_to_mlflow_skewed(context: OpExecutionContext, train_model, split_data, predict):
    logger.add("mlflow_training.log", rotation="1 MB", level="INFO")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # Get test predictions and re-index to match y_test
    y_test_pred = pd.Series(predict["y_test_pred"], index=y_test.index)

    (model, algo) = train_model
    model_name = "biggs_" + algo + "_model"

    # Apply artificial drift to simulate data skew
    skewed_data = skew_data(split_data)

    # Align predictions with the skewed data index
    y_test_pred_skewed = y_test_pred.loc[skewed_data["y_train"].index]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run started: {run_id}")

        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_params({"n_estimators": 100, "random_state": 42})

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)

        # Generate a SHAP summary plot for the original data
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("shap_summary_skewed.png", bbox_inches="tight")
        mlflow.log_artifact("shap_summary_skewed.png")
        plt.close()

        # Generate Evidently report comparing original and skewed data
        report_data = pd.DataFrame({"target": y_test, "prediction": y_test_pred})
        skewed_report_data = pd.DataFrame(
            {"target": skewed_data["y_train"], "prediction": y_test_pred_skewed}
        )
        report = Report(metrics=[RegressionPreset()])
        report.run(reference_data=report_data, current_data=skewed_report_data)
        report.save_html("evidently_report_skewed.html")
        mlflow.log_artifact("evidently_report_skewed.html")

        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{model_name}",
            name=f"{model_name}_skewed",
        )

        logger.info(f"Model registered: {result.name}, version: {result.version}")
        logger.info(f"Train Score: {train_score}, Test Score: {test_score}")


@op
def skew_data(split_data):
    """Applies artificial drift to data for regression forecasting."""
    X_train = split_data["X_train"].copy()
    y_train = split_data["y_train"].copy()

    # Apply a multiplicative drift to the first predictor (lag feature)
    X_train.iloc[:, 0] = X_train.iloc[:, 0] * 1.2
    # Add noise to predictors
    noise = np.random.normal(0, 0.1, X_train.shape)
    X_train += noise

    # Apply a drift to the target variable: increase by 10% plus some noise
    y_train = y_train * 1.1 + np.random.normal(0, 0.5, size=y_train.shape)

    return {"X_train": X_train, "y_train": y_train}


@job
def biggs_training_job():
    # First, fetch external data from the client database.
    ext_data = external_data()
    # Print the head (first 5 rows) of each DataFrame from external_data.
    print_external_data_head(ext_data)
    # Next, use the mock dataset for training/testing (for now).
    dataset = biggs_dataset()
    split = split_data(dataset)
    model = train_model(split)
    predictions = predict(model, split)
    log_to_mlflow(model, split, predictions)
    log_to_mlflow_skewed(model, split, predictions)


defs = Definitions(
    assets=[biggs_dataset, split_data],
    jobs=[biggs_training_job],
)
