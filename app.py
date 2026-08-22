import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CarPrice AI",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        linear-gradient(rgba(8, 12, 20, 0.92), rgba(8, 12, 20, 0.96)),
        radial-gradient(circle at top right, #243b55, transparent 45%),
        radial-gradient(circle at bottom left, #141e30, transparent 45%);
}

/* Main title */
.main-title {
    text-align: center;
    padding: 20px 0 5px 0;
}

.main-title h1 {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 5px;
}

.main-title p {
    font-size: 18px;
    color: #b8c1d1;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

/* Prediction */
.prediction-card {
    text-align: center;
    background: linear-gradient(
        135deg,
        rgba(30, 60, 114, 0.75),
        rgba(42, 82, 152, 0.45)
    );
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 22px;
    padding: 35px;
    margin-top: 20px;
}

.price {
    font-size: 48px;
    font-weight: 800;
    margin: 10px 0;
}

.small-text {
    color: #b8c1d1;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
}

/* Input labels */
label {
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class="main-title">
    <h1>🚗 CarPrice AI</h1>
    <p>Smart Used Car Price Prediction</p>
</div>
""", unsafe_allow_html=True)


st.markdown("---")


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = Path("used_car_price_model.pkl")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# MODEL CHECK
# =========================================================

if model is None:

    st.error(
        "❌ Model file not found. Make sure "
        "`used_car_price_model.pkl` is present in the GitHub repository."
    )

    st.stop()


# =========================================================
# AVAILABLE VALUES
# =========================================================

locations = [
    "Mumbai",
    "Pune",
    "Chennai",
    "Coimbatore",
    "Hyderabad",
    "Jaipur",
    "Kochi",
    "Kolkata",
    "Delhi",
    "Bangalore",
    "Ahmedabad"
]

fuel_types = [
    "CNG",
    "Diesel",
    "Petrol",
    "LPG",
    "Electric"
]

transmissions = [
    "Manual",
    "Automatic"
]

owner_types = [
    "First",
    "Second",
    "Third",
    "Fourth & Above"
]


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("""
<div class="card">
<h2>🚘 Car Details</h2>
<p class="small-text">
Enter the details of the used car to estimate its market price.
</p>
</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)


with col1:

    car_name = st.text_input(
        "🚘 Car Name",
        placeholder="Example: Maruti Swift Dzire"
    )

    location = st.selectbox(
        "📍 Location",
        locations
    )

    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2018,
        step=1
    )

    kilometers = st.number_input(
        "🛣️ Kilometers Driven",
        min_value=0,
        value=50000,
        step=1000
    )


with col2:

    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        fuel_types
    )

    transmission = st.selectbox(
        "⚙️ Transmission",
        transmissions
    )

    owner_type = st.selectbox(
        "👤 Owner Type",
        owner_types
    )

    mileage = st.number_input(
        "📊 Mileage (km/kg)",
        min_value=0.0,
        value=18.0,
        step=0.1
    )


with col3:

    engine = st.number_input(
        "🔧 Engine (CC)",
        min_value=500.0,
        value=1200.0,
        step=50.0
    )

    power = st.number_input(
        "⚡ Power (bhp)",
        min_value=20.0,
        value=80.0,
        step=1.0
    )

    seats = st.number_input(
        "💺 Seats",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

    new_price = st.number_input(
        "💰 New Car Price (Lakhs)",
        min_value=0.0,
        value=8.0,
        step=0.1
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_col, reset_col = st.columns(2)


with predict_col:

    predict_button = st.button(
        "🔮 Predict Used Car Price",
        type="primary"
    )


with reset_col:

    reset_button = st.button(
        "🔄 Reset"
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not car_name.strip():

        st.warning("⚠️ Please enter the car name.")

    else:

        try:

            # Create input DataFrame
            input_data = pd.DataFrame({
                "Name": [car_name],
                "Location": [location],
                "Year": [year],
                "Kilometers_Driven": [kilometers],
                "Fuel_Type": [fuel_type],
                "Transmission": [transmission],
                "Owner_Type": [owner_type],
                "Mileage": [mileage],
                "Engine": [engine],
                "Power": [power],
                "Seats": [seats],
                "New_Price": [new_price]
            })


            # Prediction
            prediction = model.predict(input_data)[0]


            # Convert prediction to lakhs
            prediction_lakhs = prediction


            # Result
            st.markdown(f"""
            <div class="prediction-card">

                <div style="font-size:20px;">
                    💰 Estimated Used Car Price
                </div>

                <div class="price">
                    ₹ {prediction_lakhs:.2f} Lakhs
                </div>

                <div class="small-text">
                    AI-powered price estimation based on the car details
                </div>

            </div>
            """, unsafe_allow_html=True)


        except Exception as e:

            st.error("❌ Prediction failed.")

            st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    color:#8892a6;
    padding:20px;
">
    🚗 CarPrice AI &nbsp; | &nbsp; Machine Learning Project
</div>
""", unsafe_allow_html=True)
