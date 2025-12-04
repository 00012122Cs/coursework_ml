"""Shared utilities for the life expectancy coursework Streamlit app."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "life_expectancy.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "life_expectancy_clean.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "final_model.pkl"
METRICS_PATH = MODELS_DIR / "model_performance.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    """Load the WHO export which is stored as a JSON payload in CSV format."""
    text = DATA_PATH.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = json.loads(text)
        records = payload.get("value", [])
        return pd.DataFrame(records)
    return pd.read_csv(DATA_PATH)


def to_snake_case(value: str) -> str:
    value = value or ""
    return "_".join(
        filter(None, "".join(ch if ch.isalnum() else " " for ch in value).lower().split())
    )


def map_gender(value: str) -> str:
    mapping = {"sex_mle": "Male", "sex_fmle": "Female", "sex_btsx": "Both sexes"}
    if isinstance(value, str):
        return mapping.get(value.lower(), value)
    return "Both sexes"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def enrich_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df.get("dim1", "SEX_BTSX").apply(map_gender)
    df["country_code"] = df.get("spatial_dim")
    df["continent_code"] = df.get("parent_location_code")
    df["continent"] = df.get("parent_location")
    df["year"] = df.get("time_dim").astype(int)
    df["life_expectancy"] = df.get("numeric_value")
    df["life_expectancy_low"] = df.get("low")
    df["life_expectancy_high"] = df.get("high")
    df["record_date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["value_range"] = df["life_expectancy_high"] - df["life_expectancy_low"]
    columns_to_drop = [
        "@odata_context",
        "dim1",
        "dim1_type",
        "dim2",
        "dim2_type",
        "dim3",
        "dim3_type",
        "time_dimension_value",
        "value",
    ]
    df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True)
    return df


def prepare_clean_dataframe() -> pd.DataFrame:
    df = enrich_columns(clean_column_names(load_raw_data()))
    df = df[df["life_expectancy"].between(0, 120)]
    df = df.drop_duplicates(subset=["country_code", "year", "gender"])
    df["continent_encoded"] = df["continent"].astype("category").cat.codes
    year_min, year_max = df["year"].min(), df["year"].max()
    df["year_normalized"] = (df["year"] - year_min) / (year_max - year_min)
    df["continent_life_expectancy_mean"] = df.groupby("continent")["life_expectancy"].transform("mean")
    df["country_life_expectancy_mean"] = df.groupby("country_code")["life_expectancy"].transform("mean")
    return df


def load_clean_data() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = prepare_clean_dataframe()
    df.to_csv(PROCESSED_PATH, index=False)
    return df


def feature_columns() -> List[str]:
    return [
        "year",
        "gender",
        "continent",
        "country_code",
        "life_expectancy_low",
        "life_expectancy_high",
        "value_range",
        "year_normalized",
        "continent_life_expectancy_mean",
        "country_life_expectancy_mean",
        "continent_encoded",
    ]


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[feature_columns()].copy()
    y = df["life_expectancy"].copy()
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "year",
        "life_expectancy_low",
        "life_expectancy_high",
        "value_range",
        "year_normalized",
        "continent_life_expectancy_mean",
        "country_life_expectancy_mean",
        "continent_encoded",
    ]
    categorical_features = ["gender", "continent", "country_code"]

    numeric_pipeline = Pipeline(
        steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def model_config() -> Dict[str, Dict[str, object]]:
    return {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {"regressor__fit_intercept": [True, False]},
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "regressor__n_estimators": [150, 250],
                "regressor__max_depth": [None, 12],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "regressor__n_estimators": [150, 200],
                "regressor__learning_rate": [0.05, 0.1],
            },
        },
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    progress_cb: Callable[[float], None] | None = None,
) -> Tuple[pd.DataFrame, str, Pipeline, List[str]]:
    records: List[Dict[str, object]] = []
    best_models: Dict[str, Pipeline] = {}
    logs: List[str] = []
    configs = model_config()
    total = len(configs)

    for idx, (name, cfg) in enumerate(configs.items(), start=1):
        logs.append(f"Training {name} with GridSearchCV ...")
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("regressor", cfg["model"])]
        )
        grid = GridSearchCV(
            pipeline,
            param_grid=cfg["params"],
            cv=5,
            n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )
        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        metrics["Model"] = name
        metrics["Best Params"] = grid.best_params_
        records.append(metrics)
        best_models[name] = grid.best_estimator_
        logs.append(
            f"{name}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}"
        )
        if progress_cb:
            progress_cb(idx / total)

    results_df = pd.DataFrame(records)
    best_row = results_df.sort_values(by="R2", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_pipeline = best_models[best_name]
    return results_df, best_name, best_pipeline, logs


def save_model(pipeline: Pipeline, metrics_df: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)


def load_trained_model() -> Pipeline | None:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def load_metrics() -> pd.DataFrame:
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()
