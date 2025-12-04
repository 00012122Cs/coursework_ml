import streamlit as st

from utils import load_clean_data, save_model, split_data, train_models

st.set_page_config(page_title="Model Training", page_icon="🤖")
st.title("🤖 Model Training")

clean_df = load_clean_data()
st.write(
    "Train three regression models (Linear Regression, Random Forest, Gradient Boosting) using"
    " the same preprocessing steps as the notebooks. Results are persisted to `models/`."
)

if st.button("Start training", type="primary"):
    progress_bar = st.progress(0.0)

    def update_progress(value: float) -> None:
        progress_bar.progress(value)

    with st.spinner("Running GridSearchCV for each model..."):
        X_train, X_test, y_train, y_test = split_data(clean_df)
        results_df, best_name, best_pipeline, logs = train_models(
            X_train, X_test, y_train, y_test, progress_cb=update_progress
        )
        save_model(best_pipeline, results_df)

    st.success(f"Training complete – best model: {best_name}")
    st.subheader("Model comparison")
    st.dataframe(results_df)

    best_params = results_df.loc[results_df["Model"] == best_name, "Best Params"].iloc[0]
    st.subheader("Best hyperparameters")
    st.json(best_params)

    st.subheader("Training logs")
    st.code("\n".join(logs))
else:
    st.info("Click the button above to trigger full training.")
