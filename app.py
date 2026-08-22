"""
VaahanValue — AI-Powered Used Car Valuation
=============================================
A Streamlit UI for an existing, already-trained used-car price model.

IMPORTANT — read this before deploying:
This file ONLY rebuilds the user interface. It does NOT retrain, refit,
or change the model in any way. All it does is:
  1. Collect inputs through dropdowns / sliders (no free-typing car names).
  2. Arrange those inputs into a single-row table that matches the
     column names your model was trained on.
  3. Call model.predict() on that table and show the result.

HOW THE INPUT TABLE IS BUILT (step 8-10 in the brief):
Most scikit-learn models remember the exact column names they were
trained on in an attribute called `feature_names_in_`. This app reads
that attribute straight from your saved model and builds the input
row to match it EXACTLY — so this app self-adapts to your model's
real preprocessing instead of guessing it. This covers both common
cases: models trained on one-hot encoded columns (e.g. "Fuel_Type_Diesel")
and models trained on the plain column names (e.g. "Fuel_Type").

If your model does NOT expose `feature_names_in_` (rare, e.g. a raw
numpy-based pipeline), the app falls back to a fixed column order that
you can edit in the `FALLBACK_FEATURE_ORDER` list below — search for
"EDIT THIS" to find every place you may need to adjust for your exact
training pipeline.
"""

import re
import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# 1. BASIC CONFIG — file paths & dataset categories
# ============================================================
MODEL_PATH = Path("used_car_price_model.pkl")
DATA_PATH = Path("used_cars_data.csv")
CURRENT_YEAR = datetime.now().year

LOCATIONS = ["Ahmedabad", "Bangalore", "Chennai", "Coimbatore", "Delhi",
             "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune"]
FUEL_TYPES = ["CNG", "Diesel", "Electric", "LPG", "Petrol"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNER_TYPES = ["First", "Second", "Third", "Fourth & Above"]
SEAT_OPTIONS = [2, 4, 5, 6, 7, 8, 9, 10]

# EDIT THIS if your model was trained on log(Price) instead of raw Price.
# Most simple tutorials train directly on Price (in Lakhs), so no
# transform is applied here by default.
PREDICTION_IS_LOG_TRANSFORMED = False

# EDIT THIS if your model has no feature_names_in_ — put your exact
# training column order here instead.
FALLBACK_FEATURE_ORDER = [
    "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats",
    "Car_Age", "Location", "Fuel_Type", "Transmission", "Owner_Type",
]

st.set_page_config(
    page_title="VaahanValue | AI Used Car Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. LOAD MODEL & CAR-NAME DATA (cached so it only runs once)
# ============================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    # A 0-byte or suspiciously tiny file usually means the .pkl is actually
    # a Git LFS pointer (common on Streamlit Cloud) rather than the real
    # model binary — catch that early with a clear message instead of a
    # raw EOFError.
    if MODEL_PATH.stat().st_size < 1024:
        st.error(
            f"⚠️ **{MODEL_PATH}** is only {MODEL_PATH.stat().st_size} bytes — "
            "that's too small to be a real model file. This usually means it's "
            "a Git LFS pointer file rather than the actual binary. Check "
            "`git lfs ls-files` in your repo, or re-upload the model without LFS."
        )
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Failed to load **{MODEL_PATH}**: {e}")
        return None


@st.cache_data
def load_brand_model_map():
    """Reads the dataset and builds {Brand: [Model names]} for the dropdowns."""
    if not DATA_PATH.exists():
        # Small safe fallback so the UI still renders without the CSV.
        return {"Maruti": ["Swift VDI"], "Hyundai": ["Creta 1.6 CRDi SX Option"]}

    df = pd.read_csv(DATA_PATH).dropna(subset=["Name"])
    df["Brand"] = df["Name"].apply(lambda n: str(n).split()[0])
    df["Model"] = df["Name"].apply(lambda n: " ".join(str(n).split()[1:]) or "Base")
    mapping = df.groupby("Brand")["Model"].apply(lambda s: sorted(set(s))).to_dict()
    return mapping


model = load_model()
brand_model_map = load_brand_model_map()
BRANDS = sorted(brand_model_map.keys())

# ============================================================
# 3. DARK AUTOMOTIVE THEME (CSS)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- background: midnight garage + carbon-fibre texture ---- */
.stApp {
    background:
        repeating-linear-gradient(45deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 2px, transparent 2px, transparent 6px),
        radial-gradient(circle at 15% 0%, rgba(242,169,59,0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 100%, rgba(0,229,255,0.06) 0%, transparent 45%),
        linear-gradient(160deg, #05070A 0%, #0D1117 45%, #10151C 100%);
    background-attachment: fixed;
}

#MainMenu, footer, header {visibility: hidden;}

/* ---- top brand strip ---- */
.plate-badge {
    display: inline-block;
    background: linear-gradient(180deg, #f5f5f0, #e2e2da);
    color: #0B0E11;
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    letter-spacing: 3px;
    padding: 6px 16px;
    border-radius: 6px;
    border: 3px solid #0B0E11;
    font-size: 0.85rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4);
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 2.6rem;
    background: linear-gradient(90deg, #F2A93B 0%, #FFD98E 45%, #00E5FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 10px 0 0 0;
    line-height: 1.1;
}
.hero-sub {
    color: #8B98A5;
    font-size: 1rem;
    margin-top: 6px;
}
hr.road {
    border: none;
    height: 3px;
    margin: 22px 0 30px 0;
    background: repeating-linear-gradient(90deg, #F2A93B 0 26px, transparent 26px 46px);
    opacity: 0.55;
    border-radius: 3px;
}

/* ---- glass section cards ---- */
.section-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 22px 26px 8px 26px;
    margin-bottom: 22px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}
.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.05rem;
    color: #F2A93B;
    letter-spacing: 1px;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}

/* ---- inputs ---- */
div[data-baseweb="select"] > div, .stSlider {
    background-color: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    color: #EDEFF2 !important;
}
.stNumberInput input {
    background-color: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    color: #EDEFF2 !important;
    caret-color: #F2A93B !important;
    cursor: text !important;
    pointer-events: auto !important;
}
/* Hide the native browser number spinner — Streamlit already renders its
   own +/- step buttons, so the native one only overlaps and blocks clicks
   into the text area. */
.stNumberInput input::-webkit-outer-spin-button,
.stNumberInput input::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}
.stNumberInput input[type=number] {
    -moz-appearance: textfield !important;
}
label, .stSlider label, .stSelectbox label, .stNumberInput label {
    color: #C6CDD6 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}

/* ---- buttons ---- */
.stButton > button, .stFormSubmitButton > button {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    border: none;
    transition: all 0.2s ease;
}
div[data-testid="stFormSubmitButton"]:nth-of-type(1) button,
.predict-btn button {
    background: linear-gradient(90deg, #F2A93B, #FF7A3D);
    color: #12100B;
    box-shadow: 0 6px 20px rgba(242,169,59,0.35);
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(242,169,59,0.45);
}

/* ---- prediction gauge card ---- */
.gauge-card {
    background: radial-gradient(circle at 50% 0%, rgba(0,229,255,0.10), rgba(255,255,255,0.02));
    border: 1px solid rgba(0,229,255,0.35);
    border-radius: 24px;
    padding: 34px 20px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,229,255,0.12), inset 0 0 30px rgba(0,229,255,0.05);
    margin-top: 6px;
}
.gauge-label {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 3px;
    color: #8FE9FF;
    font-size: 0.85rem;
}
.gauge-value {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 3.6rem;
    color: #F5FDFF;
    text-shadow: 0 0 18px rgba(0,229,255,0.65), 0 0 36px rgba(0,229,255,0.35);
    margin: 6px 0;
}
.gauge-note { color: #7C8894; font-size: 0.85rem; }

.footer-note {
    text-align: center; color: #5B6572; font-size: 0.8rem; margin-top: 40px;
}

@media (max-width: 640px) {
    .hero-title { font-size: 1.8rem; }
    .gauge-value { font-size: 2.4rem; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. HEADER
# ============================================================
st.markdown('<span class="plate-badge">IND · AI-VALUATION</span>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🚗 VaahanValue</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered used car price estimator — trained on real Indian resale listings.</div>', unsafe_allow_html=True)
st.markdown('<hr class="road">', unsafe_allow_html=True)

if model is None:
    st.error(
        f"⚠️ Couldn't find **{MODEL_PATH}**. Place your trained model file "
        "in the same folder as this app.py and refresh the page."
    )

# ============================================================
# 5. CAR IDENTITY — Brand / Model dropdowns (live, outside form
#    so the Model list updates the instant Brand changes)
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚘 Car Identity</div>', unsafe_allow_html=True)

id_col1, id_col2, id_col3 = st.columns(3)
with id_col1:
    brand = st.selectbox("Brand", BRANDS, key="brand")
with id_col2:
    models_for_brand = brand_model_map.get(brand, ["Base"])
    car_model = st.selectbox("Model", models_for_brand, key="model")
with id_col3:
    location = st.selectbox("Registered City", LOCATIONS, key="location")

full_car_name = f"{brand} {car_model}"
st.caption(f"Selected car: **{full_car_name}**")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 6. SPECIFICATIONS + USAGE FORM
# ============================================================
with st.form("car_form"):

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Specifications</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES, key="fuel_type")
    with s2:
        transmission = st.selectbox("Transmission", TRANSMISSIONS, key="transmission")
    with s3:
        owner_type = st.selectbox("Owner Type", OWNER_TYPES, key="owner_type")
    with s4:
        seats = st.selectbox("Seats", SEAT_OPTIONS, index=3, key="seats")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Usage & Performance</div>', unsafe_allow_html=True)
    u1, u2 = st.columns(2)
    with u1:
        year = st.slider("Year of Purchase", 1996, CURRENT_YEAR, 2018, key="year")
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1_000_000,
                                     value=40000, step=1000, key="km_driven")
    with u2:
        st.write("")  # small vertical alignment nudge to match the slider's height
    p1, p2, p3 = st.columns(3)
    with p1:
        mileage = st.number_input("Mileage (kmpl / km per kg)", min_value=0.0, max_value=40.0,
                                   value=18.0, step=0.1, key="mileage")
    with p2:
        engine = st.number_input("Engine (CC)", min_value=50, max_value=6000,
                                  value=1200, step=10, key="engine")
    with p3:
        power = st.number_input("Power (bhp)", min_value=20.0, max_value=800.0,
                                 value=85.0, step=1.0, key="power")
    st.markdown('</div>', unsafe_allow_html=True)

    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
    with btn_col1:
        predict_clicked = st.form_submit_button("🔮 Predict Price")
    with btn_col2:
        reset_clicked = st.form_submit_button("♻️ Reset")

# ============================================================
# 7. RESET LOGIC
# ============================================================
if reset_clicked:
    for key in ["brand", "model", "location", "fuel_type", "transmission",
                "owner_type", "seats", "year", "km_driven", "mileage",
                "engine", "power"]:
        st.session_state.pop(key, None)
    st.session_state.pop("last_prediction", None)
    st.rerun()

# ============================================================
# 8. BUILD MODEL-COMPATIBLE INPUT ROW
# ============================================================
def build_model_input(model, raw: dict) -> pd.DataFrame:
    """Builds a single-row DataFrame matching the model's expected columns.

    If the model exposes `feature_names_in_` (set automatically by
    scikit-learn when trained on a DataFrame), we align to it exactly —
    this works whether the model expects plain columns (e.g. "Fuel_Type")
    or one-hot encoded columns (e.g. "Fuel_Type_Diesel").
    """
    car_age = CURRENT_YEAR - raw["Year"]
    base_values = {
        "Name": raw["Name"],
        "Brand": raw["Brand"],
        "Location": raw["Location"],
        "Year": raw["Year"],
        "Car_Age": car_age,
        "Kilometers_Driven": raw["Kilometers_Driven"],
        "Fuel_Type": raw["Fuel_Type"],
        "Transmission": raw["Transmission"],
        "Owner_Type": raw["Owner_Type"],
        "Mileage": raw["Mileage"],
        "Engine": raw["Engine"],
        "Power": raw["Power"],
        "Seats": raw["Seats"],
    }

    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        row = {}
        for col in expected_cols:
            if col in base_values:
                row[col] = base_values[col]
                continue
            # Try to match a one-hot encoded column like "Location_Mumbai"
            matched = False
            for key, val in base_values.items():
                if col == f"{key}_{val}":
                    row[col] = 1
                    matched = True
                    break
            if not matched:
                row[col] = 0
        return pd.DataFrame([row])[expected_cols]

    # Fallback path — EDIT FALLBACK_FEATURE_ORDER above if this is used.
    row = {k: base_values[k] for k in FALLBACK_FEATURE_ORDER if k in base_values}
    return pd.DataFrame([row])


# ============================================================
# 9. PREDICT
# ============================================================
if predict_clicked and model is not None:
    raw_inputs = {
        "Name": full_car_name,
        "Brand": brand,
        "Location": location,
        "Year": year,
        "Kilometers_Driven": km_driven,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats,
    }
    try:
        model_input = build_model_input(model, raw_inputs)
        prediction = model.predict(model_input)[0]
        if PREDICTION_IS_LOG_TRANSFORMED:
            import numpy as np
            prediction = float(np.expm1(prediction))
        st.session_state["last_prediction"] = round(float(prediction), 2)
    except Exception as e:
        st.error(f"Prediction failed — the input columns may not match your model. Details: {e}")

# ============================================================
# 10. RESULT CARD
# ============================================================
if "last_prediction" in st.session_state:
    price = st.session_state["last_prediction"]
    st.markdown(f"""
    <div class="gauge-card">
        <div class="gauge-label">ESTIMATED RESALE VALUE</div>
        <div class="gauge-value">₹ {price:.2f} L</div>
        <div class="gauge-note">for {full_car_name} · {year} · {location}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer-note">Built with Streamlit · Portfolio ML Project</div>', unsafe_allow_html=True)
