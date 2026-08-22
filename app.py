
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("used_car_price_model.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CarPrice AI",
    page_icon="🚗",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

.main-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 35px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
}

.result-box {
    background-color: #1e293b;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
    margin-top: 30px;
}

.result-title {
    font-size: 20px;
    color: #cbd5e1;
}

.price {
    font-size: 48px;
    font-weight: bold;
    color: #38bdf8;
    margin: 10px;
}

.info {
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🚗 CarPrice AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Used Car Price Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# INPUT SECTION
# ==========================================

left, right = st.columns(2)


# ------------------------------------------
# LEFT COLUMN
# ------------------------------------------

with left:

    st.markdown(
        '<div class="section-title">🚘 Car Information</div>',
        unsafe_allow_html=True
    )

    st.write("")

    name = st.text_input(
        "Car Name",
        placeholder="Example: Maruti Swift Dzire VDI"
    )

    location = st.selectbox(
        "Location",
        [
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
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1998,
        max_value=2026,
        value=2018
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=1000000,
        value=50000,
        step=1000
    )

    mileage = st.number_input(
        "Mileage (km/kg or kmpl)",
        min_value=0.0,
        max_value=100.0,
        value=18.0,
        step=0.1
    )


# ------------------------------------------
# RIGHT COLUMN
# ------------------------------------------

with right:

    st.markdown(
        '<div class="section-title">⚙️ Technical Details</div>',
        unsafe_allow_html=True
    )

    st.write("")

    fuel = st.selectbox(
        "Fuel Type",
        [
            "CNG",
            "Diesel",
            "Petrol",
            "LPG",
            "Electric"
        ]
    )

    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )

    owner = st.selectbox(
        "Owner Type",
        [
            "First",
            "Second",
            "Third",
            "Fourth & Above"
        ]
    )

    engine = st.number_input(
        "Engine (CC)",
        min_value=500.0,
        max_value=6000.0,
        value=1200.0,
        step=50.0
    )

    power = st.number_input(
        "Power (bhp)",
        min_value=20.0,
        max_value=1000.0,
        value=75.0,
        step=5.0
    )

    seats = st.number_input(
        "Number of Seats",
        min_value=2.0,
        max_value=10.0,
        value=5.0,
        step=1.0
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 Predict Used Car Price",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    if name.strip() == "":
        st.warning("Please enter the car name.")

    else:

        input_data = pd.DataFrame({

            "Name": [name],

            "Location": [location],

            "Year": [year],

            "Kilometers_Driven": [km_driven],

            "Fuel_Type": [fuel],

            "Transmission": [transmission],

            "Owner_Type": [owner],

            "Mileage": [mileage],

            "Engine": [engine],

            "Power": [power],

            "Seats": [seats]

        })

        prediction = model.predict(input_data)[0]


        # ----------------------------------
        # RESULT
        # ----------------------------------

        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-title">
                    Estimated Market Price
                </div>

                <div class="price">
                    ₹ {prediction:,.2f} Lakhs
                </div>

                <div class="info">
                    Powered by Random Forest Regression
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------
        # CAR SUMMARY
        # ----------------------------------

        st.write("")

        st.subheader("📋 Car Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Year", year)
            st.metric("Fuel", fuel)

        with summary_col2:
            st.metric("KM Driven", f"{km_driven:,}")
            st.metric("Transmission", transmission)

        with summary_col3:
            st.metric("Engine", f"{engine:.0f} CC")
            st.metric("Power", f"{power:.0f} bhp")
