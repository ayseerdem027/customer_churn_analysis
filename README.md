Customer Churn Prediction

A machine learning project to predict telecom customer churn using logistic regression. Includes data preprocessing, modeling, and an interactive Streamlit app for real-time predictions.

Features
Data cleaning, encoding, and scaling

Logistic regression model training and evaluation

Saved model and preprocessor for reuse

Streamlit app for easy prediction input and results display

Installation:

git clone https://github.com/ayseerdem027/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

Usage:
Train the model:
python src/utils.py

Run the app:
py -m streamlit run app/app.py

Data:
Uses the Telco Customer Churn dataset (Kaggle). Cleaned and preprocessed data stored in data/cleaned and data/processed.

License
MIT License. See LICENSE.

