import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

from utils import feature_columns, load_clean_data

st.set_page_config(page_title="Preprocessing", page_icon="⚙️")
st.title("⚙️ Data Preprocessing")

clean_df = load_clean_data()
st.write(
    "The preprocessing pipeline enforces data quality, imputes missing values, scales numerics,"
    " and encodes categorical attributes before model training."
)

with st.expander("Cleaned dataframe", expanded=True):
    st.dataframe(clean_df.head(20))

apply_scaling = st.checkbox("Apply StandardScaler to numeric features", value=True)
apply_encoding = st.checkbox("Apply OneHotEncoder to categorical features", value=True)

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

numeric_steps = [("imputer", KNNImputer(n_neighbors=5))]
if apply_scaling:
    numeric_steps.append(("scaler", StandardScaler()))

categorical_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
if apply_encoding:
    categorical_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)))

preview_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline(steps=numeric_steps), numeric_features),
        ("categorical", Pipeline(steps=categorical_steps), categorical_features),
    ]
)

preview_features = clean_df[feature_columns()]
preview_array = preview_preprocessor.fit_transform(preview_features.head(200))
preview_cols = preview_preprocessor.get_feature_names_out()
preview_df = pd.DataFrame(preview_array, columns=preview_cols)

st.subheader("Transformed feature sample")
st.dataframe(preview_df.head(10))
st.caption("First 10 transformed rows based on the selected toggles.")

st.subheader("Preprocessing checklist")
st.markdown(
    """
- ✅ Missing values handled via `KNNImputer` (numeric) and `SimpleImputer` (categorical).
- ✅ Impossible entries filtered (life expectancy outside 0–120).
- ✅ Feature engineering: year normalisation, continent/country expectation averages.
- ✅ One-hot encoding for `gender`, `continent`, and `country_code`.
- ✅ Scaling applied to stabilise algorithms that rely on distance metrics.
"""
)
