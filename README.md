# laptop-price-predictor

A Machine Learning-based web application that predicts the price of a laptop based on user-provided specifications.

This project uses preprocessed laptop data and a trained model pipeline to estimate laptop prices using features such as brand, processor, RAM, storage, and more. Models are built using Python and machine learning libraries, and predictions are served via a simple application.

---

## Project Overview

Laptop prices can vary widely based on technical specifications like CPU, RAM, storage, display, etc. This project trains a machine learning model on historical laptop specifications to predict expected laptop prices. The model is saved using pickle and integrated into a Python application for real-time price predictions.

---

## Features

- Load dataset & preprocess data  
- Train ML models (Regression) to predict laptop price  
- Save trained model pipeline (`pipe.pkl`) and preprocessed data (`df.pkl`)  
- Serve predictions through **app.py** (Streamlit)  
- Notebook includes EDA and model training steps

## Built With

- Python
- pandas, NumPy
- scikit-learn
- Seaborn
- Jupyter Notebook
- Streamlit
- 
---
## Models Tested & Results

| Model                       | R² Score   | MAE        | Remarks                 |
| --------------------------- | ---------- | ---------- | ----------------------- |
| **Random Forest Regressor** | **0.8873** | **0.1586** | ⭐ Best Performing Model |
| Extra Trees Regressor       | 0.8850     | 0.1615     | Very strong performance |
| Gradient Boosting           | 0.8818     | 0.1595     | High accuracy           |
| Decision Tree               | 0.8455     | 0.1805     | Good but overfits       |
| Ridge Regression            | 0.8127     | 0.2092     | Stable linear model     |
| SVM (RBF Kernel)            | 0.8083     | 0.2023     | Good generalization     |
| Linear Regression           | 0.8073     | 0.2101     | Baseline linear model   |
| Lasso Regression            | 0.8071     | 0.2111     | Underperforms vs Ridge  |
| KNN Regressor               | 0.8031     | 0.1931     | Distance-based model    |
| AdaBoost Regressor          | 0.7989     | 0.2294     | Weakest model           |
