# Gender -> 1:Female 0:Male
# Churn -> 1:Yes 0:No

# What is StandardScaler?
# A tool from scikit-learn used for feature scaling.
# It standardizes numerical data so that:
# mean = 0 standard deviation = 1
# Many machine learning algorithms (like logistic regression, SVM, KNN, neural networks) work better when features are on the same scale.
# Prevents features with large values (e.g., salary in lakhs) from dominating small ones (e.g., age).

# scaler.fit_transform(X_train)
# fit() → calculates the mean and standard deviation of each column in X_train.
# transform() → uses those values to scale the data (so mean = 0, std = 1).

# joblib is a Python library used for saving and loading Python objects efficiently.
# joblib.dump(object, filename) → saves (serializes) a Python object to a file.

# Model is exported as model.pkl

# order of X: Age->Gender->Tenure->MonthlyCharges

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction")

st.divider()

st.write("please enter the value and hit the predict button to get predict")

st.divider()

age = st.number_input("Enter age: ", min_value=10, max_value=100, value=30)

gender = st.selectbox("Enter the Gender: ", ["Male", "Female"])

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthlycharge = st.number_input("Enter Monthly Charge: ", min_value=30, max_value=150)
    

st.divider()

predictbutton = st.button("Predict")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthlycharge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)

    predicted = "Yes, Churn" if prediction == 1 else "No, Not Churn"

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please the the values and then use predict button")