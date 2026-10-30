
import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("📊 Smart Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")

        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)

        st.subheader("Phone & Internet")

        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        st.subheader("Services")

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        st.subheader("Billing Information")

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.5
        )
        total = st.number_input(
            "Total Charges", min_value=0.0, value=850.5
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        response = requests.post(
            API_URL,
            json=customer,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction Result")

            if result["churn_prediction"] == "Yes":
                st.error("Customer is likely to churn.")
            else:
                st.success("Customer is unlikely to churn.")

            st.metric(
                "Churn Probability",
                f"{result['churn_probability'] * 100:.2f}%"
            )
            probability = float(result["churn_probability"])

            if probability >= 0.65:
                risk_level = "High"
            elif probability >= 0.35:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            st.write("Risk Level:", risk_level)
            prediction = result.get("churn_prediction", 0)

            if str(prediction).lower() in ["1", "yes", "true"]:
                prediction_label = "Yes"
            else:
                prediction_label = "No"

            st.write("Prediction:", prediction_label)
            st.write("Model Version:", result.get("model_version", "Not provided by API"))

        else:
            st.error(f"API Error {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as error:
        st.error(f"Could not connect to API: {error}")
