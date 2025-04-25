# 💳 Credit Risk Prediction App

This project is a Streamlit web app that predicts whether a loan applicant is a **Good** or **Bad Credit Risk** using a machine learning model trained on the **German Credit Dataset**.

##  Features

- Predict credit risk based on user inputs
- Streamlit web interface
- Random Forest Classifier with GridSearchCV
- Saved models, encoders, and scaler for reuse
- Feature importance analysis


## Model Workflow

1. **Data Preprocessing**
   - Removed unnecessary columns
   - Filled missing values
   - Label encoded categorical features

2. **Feature Engineering**
   - Created a `Credit_per_month` feature

3. **Model Training**
   - Train-test split
   - StandardScaler for scaling
   - Random Forest with hyperparameter tuning using GridSearchCV

4. **Model Saving**
   - Exported model, scaler, and encoders with `joblib`


## 📁 Project Structure
CreditRiskPredictionApp/
├── models/
│   ├── best_model.pkl             # Trained Random Forest model
│   ├── scaler.pkl                 # StandardScaler used during training
│   └── label_encoders.pkl         # LabelEncoders for categorical variables
├── streamlit_app.py               # Streamlit app for credit risk prediction
├── german_credit_data.csv         # Dataset used for training the model
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
