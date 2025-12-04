# Machine Learning & Data Analytics Coursework  
### University ID: 00012122  
### Module: 6COSC017C-n — Machine Learning and Data Analytics  
### Coursework Weight: 50%  

---

## 📌 Project Overview
This project implements an end-to-end **machine learning pipeline** using a real-world dataset taken from the **World Health Organization (WHO) Global Health Observatory**. The aim is to analyze, preprocess, model, evaluate, and deploy a predictive system focused on **Life Expectancy** across different countries, years, and demographic groups.

The coursework includes:
- A full **Exploratory Data Analysis (EDA)**
- **Data preprocessing** (cleaning, missing values, scaling, encoding)
- Training **three or more ML models**
- **Model evaluation & comparison**
- A **Streamlit multi-page web application**
- Full **reproducibility** (requirements.txt + structured notebooks)
- Version-controlled development with meaningful commits

Dataset URL (Official WHO API):  
https://ghoapi.azureedge.net/api/WHOSIS_000001?$format=csv

---

## 📂 Repository Structure

```
coursework_ml/
├── data/
│   ├── life_expectancy.csv          # Raw WHO API export (JSON payload)
│   └── processed/                   # Automatically created for cleaned datasets
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_preprocessing.ipynb       # Data cleaning, feature engineering, exports
│   └── 03_model_training.ipynb      # Pipelines, tuning, evaluation, model saving
├── streamlit_app/
│   ├── Home.py                      # Landing page
│   ├── utils.py                     # Shared preprocessing/model utilities
│   └── pages/
│       ├── 1_📊_EDA.py
│       ├── 2_⚙️_Preprocessing.py
│       ├── 3_🤖_Model_Training.py
│       └── 4_📈_Evaluation.py
├── models/                          # Saved pipeline + metrics (generated after training)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Open the notebooks**
   - Launch Jupyter Lab/Notebook and run the files in `notebooks/` sequentially:
     1. `01_eda.ipynb`
     2. `02_preprocessing.ipynb`
     3. `03_model_training.ipynb`
   - These notebooks reproduce the full coursework pipeline, export processed datasets to `data/processed/`, and save the best model to `models/final_model.pkl`.

3. **Run the Streamlit dashboard**
   ```bash
   streamlit run streamlit_app/Home.py
   ```
   - Navigate across the sidebar pages (EDA → Preprocessing → Model Training → Evaluation).
   - The Streamlit app reuses the same preprocessing code, lets you re-train models interactively, and supports batch prediction via CSV upload.

---

## 🧪 Deliverables

- **Data**: `data/life_expectancy.csv` contains the WHO indicator `WHOSIS_000001`.
- **Notebooks**: Document EDA, preprocessing with feature engineering, and model development with MAE/RMSE/R² comparisons plus GridSearchCV tuning.
- **Models**: Best-performing pipeline persisted as `models/final_model.pkl` together with `models/model_performance.csv`.
- **App**: Multi-page Streamlit experience for analysis, preprocessing inspection, training, and evaluation/deployment.

Follow the notebooks and app to regenerate every artefact and align with WIUT coursework requirements.
