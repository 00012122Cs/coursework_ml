import streamlit as st
import plotly.express as px

from utils import load_clean_data

st.set_page_config(page_title="EDA", page_icon="📊")
st.title("📊 Exploratory Data Analysis")

clean_df = load_clean_data()
st.write("The cleaned dataset below powers the full pipeline. Use the widgets to explore key patterns.")

with st.expander("Sample records", expanded=True):
    st.dataframe(clean_df.head(10))

st.subheader("Summary statistics")
st.dataframe(clean_df.describe(include="all").transpose())

numeric_cols = [col for col in clean_df.select_dtypes(include=["float64", "int64"]).columns if col != "life_expectancy"]
categorical_cols = [col for col in ["gender", "continent", "country_code"] if col in clean_df.columns]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### Numerical distribution")
    selected_num = st.selectbox(
        "Select numeric column", options=["life_expectancy"] + numeric_cols, index=0
    )
    fig_num = px.histogram(
        clean_df,
        x=selected_num,
        nbins=40,
        marginal="rug",
        title=f"Distribution of {selected_num}",
        opacity=0.8,
    )
    st.plotly_chart(fig_num, use_container_width=True)

with col_b:
    st.markdown("#### Categorical distribution")
    selected_cat = st.selectbox("Select categorical column", options=categorical_cols)
    counts = clean_df[selected_cat].value_counts().reset_index()
    counts.columns = [selected_cat, "count"]
    fig_cat = px.bar(counts, x=selected_cat, y="count", title=f"{selected_cat.title()} frequency")
    fig_cat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)

st.subheader("Correlation heatmap")
correlation_cols = [
    col
    for col in [
        "life_expectancy",
        "life_expectancy_low",
        "life_expectancy_high",
        "value_range",
        "year",
        "year_normalized",
    ]
    if col in clean_df.columns
]
correlation_matrix = clean_df[correlation_cols].corr()
fig_corr = px.imshow(
    correlation_matrix,
    text_auto=True,
    color_continuous_scale="Viridis",
    title="Relationship between numerical features",
)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Country and year level insights")
country_metric = clean_df.groupby("country_code")["life_expectancy"].mean().reset_index()
fig_map = px.choropleth(
    country_metric,
    locations="country_code",
    color="life_expectancy",
    color_continuous_scale="plasma",
    title="Average life expectancy by country",
)
st.plotly_chart(fig_map, use_container_width=True)

yearly_metric = clean_df.groupby("year")["life_expectancy"].mean().reset_index()
fig_year = px.line(yearly_metric, x="year", y="life_expectancy", markers=True)
fig_year.update_layout(title="Global trend across years")
st.plotly_chart(fig_year, use_container_width=True)
