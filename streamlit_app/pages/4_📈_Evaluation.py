import io

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import feature_columns, load_metrics, load_trained_model

st.set_page_config(page_title="Evaluation", page_icon="📈")
st.title("📈 Model Evaluation & Inference")

metrics_df = load_metrics()
if metrics_df.empty:
    st.warning("No evaluation results found. Train models first on the previous page.")
else:
    st.subheader("Comparison table")
    st.dataframe(metrics_df)

    metric_choice = st.selectbox("Select metric", options=["MAE", "RMSE", "R2"], index=2)
    fig = px.bar(
        metrics_df,
        x="Model",
        y=metric_choice,
        title=f"Model comparison using {metric_choice}",
        color="Model",
        text_auto=".2f",
    )
    st.plotly_chart(fig, use_container_width=True)

    best_idx = metrics_df["R2"].idxmax()
    best_row = metrics_df.loc[best_idx]
    col1, col2, col3 = st.columns(3)
    col1.metric("Best model", best_row["Model"])
    col2.metric("MAE", f"{best_row['MAE']:.2f}")
    col3.metric("RMSE", f"{best_row['RMSE']:.2f}")

st.subheader("Predict on new data")
trained_model = load_trained_model()
if trained_model is None:
    st.error("Trained model not found. Please run the training step first.")
else:
    uploaded = st.file_uploader(
        "Upload CSV with feature columns", type="csv", help=", ".join(feature_columns())
    )
    if uploaded is not None:
        df_new = pd.read_csv(uploaded)
        missing_cols = [col for col in feature_columns() if col not in df_new.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
        else:
            predictions = trained_model.predict(df_new[feature_columns()])
            result_df = df_new.copy()
            result_df["predicted_life_expectancy"] = predictions
            st.success("Predictions generated")
            st.dataframe(result_df.head(20))
            buffer = io.StringIO()
            result_df.to_csv(buffer, index=False)
            st.download_button(
                "Download predictions",
                data=buffer.getvalue(),
                file_name="life_expectancy_predictions.csv",
                mime="text/csv",
            )

st.info(
    "The evaluation view leverages `models/model_performance.csv` and the saved"
    " pipeline in `models/final_model.pkl`. Re-run training any time you change"
    " preprocessing or feature engineering choices."
)
