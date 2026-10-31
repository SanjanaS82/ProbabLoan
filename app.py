
import streamlit as st
import pickle
import numpy as np

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #a1c9d6;
}

/* Main application text */
.stApp {
    color: #17365d !important;
}

/* All headings */
h1, h2, h3, h4, h5, h6 {
    color: #17365d !important;
}

/* Regular text and descriptions */
p, label, span, li {
    color: #24445e;
}

/* Input field labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stTextArea label {
    color: #24445e !important;
    font-weight: 600 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #17365d !important;
}

/* Placeholder text */
.stTextInput input::placeholder {
    color: #8495a5 !important;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #08303d;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #365863;
    font-size: 16px;
    margin-bottom: 30px;
}

div.stButton > button {
    background-color: #1792b3;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #12708a;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #69b6cf;
}

/* Input field backgrounds */
.stTextInput input,
.stNumberInput input {
    background-color: #F5FAFC !important;
    color: #17365d !important;
    border: 1px solid #8BBACB !important;
}

/* Number input buttons */
.stNumberInput button {
    background-color: #E2F0F5 !important;
    color: #17365d !important;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------
# LOAD TRAINED MODEL
# ----------------------------------

try:
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)

except FileNotFoundError:
    st.error("Model file 'loan_model.pkl' not found!")
    st.stop()


# ----------------------------------
# HEADER
# ----------------------------------

st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart Loan Eligibility Assessment System</div>',
    unsafe_allow_html=True
)

st.divider()

st.write(
    "Enter the applicant details below to evaluate "
    "their loan approval eligibility."
)


# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:
    st.title("🏦 LoanAssist")

    st.write("Loan Eligibility Predictor")

    st.divider()

    st.subheader("About the Application")

    st.write(
        "This application uses a Machine Learning model "
        "to predict whether a loan application is likely "
        "to be approved or rejected."
    )

    st.divider()

    st.caption("Machine Learning Based Prediction System")


# ----------------------------------
# PERSONAL DETAILS
# ----------------------------------

st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input(
        "Applicant Name",
        placeholder="Enter full name"
    )

with col2:
    city = st.text_input(
        "City",
        placeholder="Enter city"
    )


st.divider()


# ----------------------------------
# FINANCIAL DETAILS
# ----------------------------------

st.subheader("📊 Financial Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        value=50000,
        step=1000
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

    years_employed = st.number_input(
        "Years Employed",
        min_value=0,
        max_value=50,
        value=5
    )

with col2:
    loan_amount = st.number_input(
        "Requested Loan Amount ($)",
        min_value=0,
        value=20000,
        step=500
    )

    points = st.number_input(
        "Credit Points / Credit Rating",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.5
    )


st.divider()


# ----------------------------------
# PREDICTION
# ----------------------------------

st.subheader("🔍 Loan Eligibility Result")

if st.button("Predict Loan Status", type="primary"):

    if not name.strip():
        st.warning("Please enter the applicant's name.")

    elif income <= 0 or loan_amount <= 0:
        st.warning("Income and loan amount must be greater than zero.")

    else:

        # Features must follow the EXACT order
        # used during model training.

        input_features = np.array([[
            income,
            credit_score,
            loan_amount,
            years_employed,
        ]])

        # Generate prediction
        prediction = model.predict(input_features)

        st.divider()

        st.subheader("📋 Prediction Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Applicant", name)

        with col2:
            st.metric("Requested Loan", f"${loan_amount:,.2f}")

        if prediction[0] == 1:

            st.success(
                f"🎉 Congratulations, {name}! "
                "Your loan application is APPROVED."
            )

            st.balloons()

        else:

            st.error(
                f"❌ Sorry, {name}. "
                "Your loan application is REJECTED "
                "based on the provided parameters."
            )

        st.caption(
            "Note: This is a machine learning prediction "
            "and does not represent an actual lending decision."
        )


# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.markdown(
    "<center>LoanAssist | Machine Learning Project</center>",
    unsafe_allow_html=True
)
