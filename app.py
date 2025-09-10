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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- LOAD ARTIFACTS ---
# Note: Ensure these files are in the same directory as your app.py
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Error: `scaler.pkl` or `model.pkl` not found. Please ensure they are in the correct directory.")
    st.stop()


# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding: 2rem;
        border-radius: 20px;
        background-color: #f0f2f6;
    }

    /* Title styling */
    h1 {
        color: #1e3a8a; /* Dark blue */
        text-align: center;
        font-weight: bold;
    }

    /* Subheader/instruction styling */
    .st-emotion-cache-1629p8f e1nzilvr5 {
        text-align: center;
        color: #4b5563; /* Gray */
    }

    /* Input widgets styling */
    .stNumberInput, .stSelectbox {
        border-radius: 10px;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: 2px solid #1e3a8a;
        background-color: #1e3a8a;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: white;
        color: #1e3a8a;
    }

    /* Prediction result card styling */
    .churn-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 2rem;
    }
    .churn-yes {
        color: #dc2626; /* Red */
        font-size: 2rem;
        font-weight: bold;
    }
    .churn-no {
        color: #16a34a; /* Green */
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- APP LAYOUT ---

st.title("Customer Churn Predictor 🤖")
st.write("Provide the customer details below to predict the likelihood of churn.")

st.divider()

# --- INPUT FORM ---
with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, help="Enter the customer's age.")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=150, value=12, step=1, help="How many months has the customer been with us?")

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the customer's gender.")
        monthly_charge = st.number_input("Monthly Charge ($)", min_value=10.0, max_value=200.0, value=75.50, step=0.5, help="Enter the customer's monthly subscription fee.")

    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict Churn")


# --- PREDICTION LOGIC ---
if submit_button:
    # 1. Preprocess the inputs
    gender_numeric = 1 if gender == "Female" else 0
    
    # 2. Create the feature vector
    features = np.array([[age, gender_numeric, tenure, monthly_charge]])
    
    # 3. Scale the features
    scaled_features = scaler.transform(features)
    
    # 4. Make a prediction
    prediction = model.predict(scaled_features)
    predicted = "Yes, Churn" if prediction == 1 else "No, Not Churn"

    # 5. Display the result
    st.divider()
    
    if prediction[0] == 1:
        st.markdown(
            f"""
            <div class="churn-card">
                <h3>Prediction Result</h3>
                <p class="churn-yes">Yes, Customer is Likely to Churn</p>
                <p>Confidence: <strong>{predicted}%</strong></p>
                <small>Consider taking retention actions for this customer.</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="churn-card">
                <h3>Prediction Result</h3>
                <p class="churn-no">No, Customer is Unlikely to Churn</p>
                <p>Confidence: <strong>{predicted}%</strong></p>
                <small>This customer appears to be loyal.</small>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Please fill in the details above and click 'Predict Churn' to see the result.")