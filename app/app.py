
from pathlib import Path
import base64
import json

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIG & BRAND
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

MODEL_PATH = BASE_DIR / "artifacts" / "flight_price_pipeline.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "flight_price_model.meta.json"


def _data_uri(mime: str, name: str) -> str:
    """Inline a file from static/ as a data URI.

    The brand mark is embedded rather than linked so the favicon resolves
    correctly on every host, whatever the public URL turns out to be.
    """
    path = STATIC_DIR / name
    if not path.exists():
        return ""
    payload = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{payload}"


MARK_URI = _data_uri("image/svg+xml", "favicon.svg") or "✈️"

st.set_page_config(
    page_title="SkyPrice — AI Flight Price Predictor",
    page_icon=MARK_URI,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Link-preview tags. The image path is relative on purpose so it resolves
# against whatever host ends up serving the app (no hard-coded domain).
# Streamlit mounts the app's ./static folder at /app/static/.
OG_IMAGE = "/app/static/og_image.png"
OG_TITLE = "SkyPrice — AI Flight Price Predictor"
OG_DESC = (
    "Estimate flight fares with a machine-learning model trained on "
    "300,153 real routes. Instant, explainable fare estimates."
)

st.markdown(
    f"""
<meta property="og:title" content="{OG_TITLE}" />
<meta property="og:description" content="{OG_DESC}" />
<meta property="og:image" content="{OG_IMAGE}" />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="{OG_TITLE}" />
<meta name="twitter:description" content="{OG_DESC}" />
<meta name="twitter:image" content="{OG_IMAGE}" />
""",
    unsafe_allow_html=True,
)


# =========================================================
# 2. MODEL
# =========================================================


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata():
    if not METADATA_PATH.exists():
        return {}

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    metadata = load_metadata()

except FileNotFoundError as error:
    st.error(f"Model or metadata file not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Error loading model: {error}")
    st.stop()


# =========================================================
# 3. DEFAULT OPTIONS
# =========================================================

AIRLINES = [
    "AirAsia",
    "Air_India",
    "GO_FIRST",
    "Indigo",
    "SpiceJet",
    "Vistara",
]

CITIES = [
    "Bangalore",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "Mumbai",
]

TIMES = [
    "Early_Morning",
    "Morning",
    "Afternoon",
    "Evening",
    "Night",
    "Late_Night",
]

STOPS = [
    "zero",
    "one",
    "two_or_more",
]

CLASSES = [
    "Economy",
    "Business",
]


# =========================================================
# 4. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

.stApp {
    background: #0B1120;
    color: #F8FAFC;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    padding: 25px 0 20px 0;
    display: flex;
    align-items: center;
    gap: 22px;
}

.hero-mark {
    width: 74px;
    height: 74px;
    border-radius: 18px;
    flex: none;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.45);
}

.hero-text {
    min-width: 0;
}

.eyebrow {
    color: #60A5FA;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #F8FAFC;
    font-size: clamp(30px, 5vw, 48px);
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 12px 0;
}

.hero p {
    color: #94A3B8;
    font-size: 16px;
}

.section-title {
    color: #F1F5F9;
    font-size: 22px;
    font-weight: 700;
}

.section-subtitle {
    color: #94A3B8;
    font-size: 13px;
    margin-bottom: 20px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111827;
    border: 1px solid #273449;
    border-radius: 16px;
}

div[data-testid="stForm"] {
    border: none;
    padding: 0;
}

h3, h4, h5 {
    color: #E2E8F0 !important;
}

label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] > div {
    background: #1E293B;
    border-color: #334155;
    border-radius: 9px;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #1E293B;
    color: #F8FAFC;
    border-radius: 9px;
}

.stSlider [data-testid="stTickBar"] {
    background: #334155;
}

.stFormSubmitButton button {
    background: linear-gradient(90deg, #2563EB, #4F46E5);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stFormSubmitButton button:hover {
    background: linear-gradient(90deg, #3B82F6, #6366F1);
    color: white;
    border: none;
    transform: translateY(-1px);
}

.result-card {
    background: linear-gradient(135deg, #172554, #1E1B4B);
    border: 1px solid #3730A3;
    border-radius: 16px;
    padding: 28px;
    margin-top: 15px;
}

.result-label {
    color: #BFDBFE;
    font-size: 14px;
}

.result-price {
    color: #FFFFFF;
    font-size: clamp(30px, 5vw, 42px);
    font-weight: 800;
    margin: 10px 0;
}

.result-note {
    color: #C7D2FE;
    font-size: 13px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #273449;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

hr {
    border-color: #273449;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. HEADER
# =========================================================

st.markdown(f"""
<div class="hero">
    <img class="hero-mark" src="{MARK_URI}" alt="SkyPrice" />
    <div class="hero-text">
        <div class="eyebrow">SkyPrice &middot; AI Flight Analytics</div>
        <h1>Flight Price Prediction</h1>
        <p>
            Discover your estimated flight fare using
            machine learning.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# 6. MODEL INPUT CONFIGURATION
# =========================================================

# Use the features from the updated metadata
MODEL_COLUMNS = metadata.get("features", [])

# if not MODEL_COLUMNS:
#     st.error("No features found in metadata.")
#     st.stop()

# st.write("Model features:", MODEL_COLUMNS)


# =========================================================
# 7. PAGE INTRO
# =========================================================

st.markdown(
    '<div class="section-title">✈ Flight information</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Fill in the details below to estimate your ticket price.'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 8. FLIGHT FORM
# =========================================================

with st.container(border=True):

    with st.form("flight_prediction_form"):

        # -------------------------
        # AIRLINE
        # -------------------------

        st.markdown("#### 01 — Airline & flight")

        col1, col2 = st.columns(2)

        with col1:
            airline = st.selectbox(
                "Airline",
                AIRLINES,
                index=0,
            )


        st.write("")

        # -------------------------
        # ROUTE
        # -------------------------

        st.markdown("#### 02 — Your route")

        col3, col4 = st.columns(2)

        with col3:
            source_city = st.selectbox(
                "Departure city",
                CITIES,
                index=CITIES.index("Delhi"),
            )

        with col4:
            destination_city = st.selectbox(
                "Destination city",
                CITIES,
                index=CITIES.index("Mumbai"),
            )

        st.write("")

        # -------------------------
        # SCHEDULE
        # -------------------------

        st.markdown("#### 03 — Flight schedule")

        col5, col6 = st.columns(2)

        with col5:
            departure_time = st.selectbox(
                "Departure time",
                TIMES,
                index=2,
            )

        with col6:
            arrival_time = st.selectbox(
                "Arrival time",
                TIMES,
                index=3,
            )

        st.write("")

        # -------------------------
        # TRAVEL DETAILS
        # -------------------------

        st.markdown("#### 04 — Travel details")

        col7, col8, col9 = st.columns(3)

        with col7:
            stops = st.selectbox(
                "Total stops",
                STOPS,
                index=1,
                format_func=lambda x: {
                    "zero": "Non-stop",
                    "one": "1 stop",
                    "two_or_more": "2+ stops",
                }[x],
            )

        with col8:
            travel_class = st.selectbox(
                "Travel class",
                CLASSES,
                index=0,
            )

        with col9:
            duration = st.number_input(
                "Duration (hours)",
                min_value=0.5,
                max_value=50.0,
                value=2.5,
                step=0.5,
            )

        days_left = st.slider(
            "Days left before departure",
            min_value=1,
            max_value=60,
            value=15,
            help="Number of days between booking and departure.",
        )

        st.write("")

        submitted = st.form_submit_button(
            "✨ Predict Flight Price",
            use_container_width=True,
        )


# =========================================================
# 9. PREDICTION
# =========================================================

if submitted:

    if source_city == destination_city:
        st.error(
            "Departure city and destination city must be different."
        )

    else:

        # Build the input using the common training feature names.
        input_values = {
            "airline": airline,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": travel_class,
            "duration": duration,
            "days_left": days_left,
        }

        try:
            # Preserve the model's expected column order.
            missing = [
                col for col in MODEL_COLUMNS
                if col not in input_values
            ]

            if missing:
                st.error(
                    "Some required model features are missing: "
                    + ", ".join(missing)
                )
                st.stop()

            input_df = pd.DataFrame(
                [[input_values[col] for col in MODEL_COLUMNS]],
                columns=MODEL_COLUMNS,
            )

            with st.spinner("Analyzing your flight..."):

                prediction = float(model.predict(input_df)[0])

            if prediction < 0:
                st.warning(
                    "The model returned a negative fare. "
                    "Please verify the input and model."
                )

            else:
                st.write("")
                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '🎯 Prediction result'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            ESTIMATED FLIGHT PRICE
                        </div>
                        <div class="result-price">
                            ₹ {prediction:,.2f}
                        </div>
                        <div class="result-note">
                            AI-generated estimate · INR
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                # -------------------------
                # TRIP SUMMARY
                # -------------------------

                st.markdown("#### Your trip summary")

                m1, m2, m3 = st.columns(3)

                m1.metric("Airline", airline)
                m2.metric("Duration", f"{duration:g} hours")
                m3.metric("Days left", f"{days_left} days")

                st.caption(
                    f"Route: {source_city} → {destination_city}"
                )

                st.caption(
                    "This is a machine learning estimate, "
                    "not a live airline quotation."
                )

        except Exception as error:
            st.error(
                "Prediction failed. Check that the input "
                "features and categories match the model "
                "training pipeline."
            )

            st.exception(error)


# =========================================================
# 10. FOOTER
# =========================================================

st.write("")
st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748B;
        font-size:12px;
        padding:10px;
    ">
        SkyPrice · Machine Learning Project · YouCode
        <br>
        Built with Streamlit & Python
    </div>
    """,
    unsafe_allow_html=True,
)