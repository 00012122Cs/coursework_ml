import streamlit as st

from utils import load_clean_data, load_raw_data

st.set_page_config(page_title="Life Expectancy Prediction", page_icon="🌍", layout="wide")

st.title("🌍 Life Expectancy Prediction Coursework")
st.markdown(
    """This dashboard accompanies the WIUT Machine Learning coursework. It summarises the
    exploratory analysis, preprocessing logic, modelling results, and deployment-ready inference
    tools built around the WHO Global Health Observatory life expectancy dataset."""
)

raw_df = load_raw_data()
clean_df = load_clean_data()

col1, col2, col3 = st.columns(3)
col1.metric("Raw records", f"{len(raw_df):,}")
col2.metric("Clean observations", f"{len(clean_df):,}")
col3.metric("Countries", clean_df["country_code"].nunique())

st.subheader("📁 Dataset Description")
st.write(
    "The dataset originates from the WHO API (indicator `WHOSIS_000001`). Each record contains a"
    " life expectancy estimate for a specific country, year, and sex, accompanied by bounds"
    " detailing statistical confidence."
)
st.markdown(
    "Key columns include `country_code`, `year`, `gender`, `life_expectancy`, and its"
    " lower/upper confidence limits. Region metadata (WHO parent locations) is leveraged for"
    " feature engineering."
)

st.subheader("🧭 Project Overview")
st.markdown(
    """The coursework delivers the full ML workflow: exploratory data analysis, preprocessing,
    model training with hyperparameter tuning, evaluation, and deployment via Streamlit.
    Each section of the site mirrors a Jupyter notebook step for transparency and reproducibility."""
)

st.info(
    "Use the sidebar navigation to move through EDA, preprocessing, model training, and"
    " evaluation pages. Notebook code lives in `notebooks/`, while the trained pipeline and"
    " processed datasets are stored in `data/processed/` and `models/`."
)
