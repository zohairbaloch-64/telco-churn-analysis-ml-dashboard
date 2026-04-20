# Telco Customer Churn Prediction

A Streamlit-based churn analytics dashboard and prediction app built with Python and scikit-learn. This project explores the Telco Customer Churn dataset, visualizes customer behavior, and predicts churn risk using a Random Forest classifier.

## What is Churn?

Churn, also known as customer attrition, refers to the rate at which customers stop doing business with a company or service provider. In the telecommunications industry, churn occurs when customers cancel their subscriptions or switch to a competitor. High churn rates can significantly impact a company's revenue and growth, making churn prediction a critical area of analytics. This project analyzes factors influencing churn and builds a predictive model to identify at-risk customers, enabling proactive retention strategies.

## Project Contents

- `app.py` - Streamlit dashboard with interactive filters, charts, model training, and churn prediction.
- `Telco_customer_churn.ipynb` - Jupyter notebook for exploratory data analysis and modeling experiments.
- `Telco-Customer-Churn.csv` - Dataset containing customer account and churn information.

## Features

- Data cleaning and preprocessing for the Telco churn dataset
- Interactive dashboard with filters for tenure and contract type
- Churn insights and visualizations using Plotly
- Model training using a Random Forest classifier
- Feature importance display for churn prediction drivers
- Live prediction interface with aligned feature encoding and safe inference
- Download filtered data from the dashboard

## Setup

1. Clone or copy the project folder.
2. Install dependencies in a Python environment:

```bash
pip install streamlit pandas scikit-learn plotly
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

- Use the sidebar filters to slice the dataset by tenure and contract type.
- Review the key churn insights and visual charts.
- Enter customer details in the prediction section and click **Predict Churn**.
- Download the filtered dataset using the download button.

## Notes

- The model trains on the same dataset at app runtime, so the prediction pipeline is aligned with the current preprocessing logic.
- Categorical features are encoded consistently between training and inference.

## Dataset

The dataset includes customer demographics, account information, services, and churn labels. It is commonly used for churn modeling and customer retention analysis.

## License

This project is intended for educational and portfolio use.
