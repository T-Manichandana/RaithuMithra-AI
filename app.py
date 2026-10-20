from pathlib import Path
import html

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "crop_model.joblib"
LEAF_MODEL_PATH = APP_DIR / "models" / "leaf_health_model.keras"

st.set_page_config(
    page_title="RaithuMithra AI | Smart Farming Companion",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Nature-inspired palette: forest green, leaf green, cream, and warm white.
st.markdown(
    r"""
    <style>
    :root {
      --forest: #214d35;
      --forest-deep: #173b29;
      --leaf: #3e8051;
      --leaf-light: #e8f3e5;
      --cream: #f7f8ef;
      --paper: #ffffff;
      --ink: #24372b;
      --muted: #5c7062;
      --border: #d7e4d4;
      --gold: #d9a441;
    }

    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background: var(--cream) !important;
      color: var(--ink) !important;
    }
    [data-testid="stHeader"] { background: rgba(247,248,239,.94) !important; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stMainBlockContainer"] { padding-top: 3rem !important; padding-bottom: 3rem; max-width: 1500px; overflow: visible !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #e5f0df 0%, #edf4e7 55%, #e1eddc 100%) !important;
      border-right: 1px solid #d1e1cc;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color: var(--forest-deep) !important; }
    .brand-lockup { padding: .4rem .25rem 1.1rem; }
    .brand-name { font-size: 1.28rem; font-weight: 800; color: var(--forest-deep); letter-spacing: -.025em; }
    .brand-tagline { color: #597260; font-size: .9rem; margin-top: .35rem; }
    .sidebar-foot {
      margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #d1e1cc;
      color: #58715e; font-size: .82rem; line-height: 1.5;
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .4rem !important; }
    /* Force all radio navigation items to use the green-and-cream theme. */
    [data-testid="stSidebar"] [role="radiogroup"] label {
      display: flex !important; align-items: center !important;
      background: #f8fbf5 !important; background-color: #f8fbf5 !important;
      border: 1px solid #d5e4cf !important;
      padding: .62rem .7rem !important; border-radius: 11px !important; transition: all .15s ease;
      color: #173b29 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label > div,
    [data-testid="stSidebar"] [role="radiogroup"] label > div > div,
    [data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {
      background: transparent !important; background-color: transparent !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
      background: #f8fbf5 !important; border-color: #a8c9a1 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
    [data-testid="stSidebar"] [role="radiogroup"] label:has([role="radio"][aria-checked="true"]) {
      background: #2f7045 !important; border-color: #2f7045 !important;
      box-shadow: 0 2px 7px rgba(33,77,53,.12);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) *,
    [data-testid="stSidebar"] [role="radiogroup"] label:has([role="radio"][aria-checked="true"]) * { color: #ffffff !important; }
    [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"],
    [data-testid="stSidebar"] [role="radiogroup"] [role="radio"] { accent-color: #2f7045 !important; color: #2f7045 !important; }
    [data-testid="stSidebar"] [role="radiogroup"] svg { color: #3e8051 !important; fill: #3e8051 !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) svg,
    [data-testid="stSidebar"] [role="radiogroup"] label:has([role="radio"][aria-checked="true"]) svg { color: #ffffff !important; fill: #ffffff !important; }

    /* Headings and typography */
    h1, h2, h3, h4 { color: var(--forest) !important; letter-spacing: -.025em; }
    h1 { font-weight: 800 !important; }
    p, li, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
    .eyebrow { display: block; overflow: visible !important; text-transform: uppercase; font-size: .76rem; font-weight: 800; line-height: 1.6 !important; letter-spacing: .12em; color: var(--leaf); margin-top: .15rem; margin-bottom: .5rem; padding-top: .1rem; }
    [data-testid="stMarkdown"]:has(.eyebrow), [data-testid="stMarkdownContainer"]:has(.eyebrow) { overflow: visible !important; min-height: 1.5rem; }
    .page-intro { color: var(--muted); font-size: 1.02rem; line-height: 1.6; margin-top: 0 !important; margin-bottom: 1.35rem; }

    /* Home hero */
    .hero {
      position: relative; overflow: hidden; padding: 2.15rem 2.2rem;
      background: linear-gradient(118deg, #e0efda 0%, #f5f8e9 64%, #eaf2d9 100%);
      border: 1px solid #cfe0c8; border-radius: 24px; margin-bottom: 1.5rem;
      box-shadow: 0 8px 24px rgba(33,77,53,.055);
    }
    .hero:after { content: "🌿"; position: absolute; right: 2rem; bottom: -.8rem; font-size: 7rem; opacity: .17; transform: rotate(-18deg); }
    .hero h1 { font-size: clamp(2rem, 3.5vw, 3rem); line-height: 1.12; margin: .3rem 0 .8rem; color: var(--forest-deep) !important; }
    .hero p { font-size: 1.08rem; line-height: 1.7; color: #415b48 !important; max-width: 760px; margin-bottom: .6rem; }
    .hero-pill { display: inline-block; border-radius: 999px; background: #ffffffb8; border: 1px solid #c9ddc3; padding: .35rem .75rem; color: var(--forest); font-size: .8rem; font-weight: 700; }

    /* Cards */
    .feature-card {
      height: 100%; min-height: 245px; padding: 1.25rem 1.25rem 1.1rem;
      background: var(--paper); border: 1px solid var(--border); border-radius: 18px;
      box-shadow: 0 5px 18px rgba(36,55,43,.045);
    }
    .feature-icon { width: 48px; height: 48px; display:flex; align-items:center; justify-content:center; border-radius: 15px; background: var(--leaf-light); font-size: 1.65rem; margin-bottom: .85rem; }
    .feature-card h3 { margin: 0 0 .5rem; font-size: 1.13rem; color: var(--forest) !important; }
    .feature-card p { color: var(--muted) !important; font-size: .92rem; line-height: 1.55; margin: 0; }
    .section-heading { margin-top: 1.3rem; margin-bottom: .8rem; }
    .soft-panel { background: #eef5e9; border: 1px solid #d7e5d0; padding: 1rem 1.1rem; border-radius: 14px; }

    /* Home page information cards */
    .home-info-card {
      height: 100%; min-height: 205px; box-sizing: border-box;
      padding: 1.35rem 1.35rem 1.25rem;
      background: #ffffff; border: 1px solid var(--border);
      border-radius: 18px; box-shadow: 0 5px 18px rgba(36,55,43,.045);
    }
    .home-info-card h3 {
      margin: .55rem 0 .55rem; font-size: 1.12rem;
      color: var(--forest) !important;
    }
    .home-info-card p {
      margin: 0; color: var(--muted) !important;
      font-size: .93rem; line-height: 1.65;
    }
    .home-card-icon {
      width: 44px; height: 44px; display: flex; align-items: center;
      justify-content: center; border-radius: 13px;
      background: var(--leaf-light); font-size: 1.45rem;
    }
    .home-bottom-heading { margin: 2rem 0 .85rem; }
    .home-note {
      margin-top: 1rem; padding: 1rem 1.15rem;
      background: #eef5e9; border: 1px solid #d7e5d0;
      border-left: 4px solid #3e8051; border-radius: 12px;
      color: #385542; font-size: .92rem; line-height: 1.6;
    }

    /* Buttons */
    div.stButton > button, div.stFormSubmitButton > button, a[kind="primary"], [data-testid="stLinkButton"] a, a[data-testid="stLinkButton"] {
      background: var(--leaf) !important; color: #ffffff !important; border: 1px solid var(--leaf) !important;
      border-radius: 11px !important; min-height: 2.8rem; font-weight: 700 !important;
      box-shadow: 0 3px 8px rgba(33,77,53,.10); transition: background .15s ease, transform .15s ease;
    }
    div.stButton > button p, div.stFormSubmitButton > button p, [data-testid="stLinkButton"] a p { color: #ffffff !important; font-weight: 700 !important; }
    [data-testid="stLinkButton"] a, a[data-testid="stLinkButton"] { background: #3e8051 !important; color: #ffffff !important; border: 1px solid #3e8051 !important; border-radius: 11px !important; font-weight: 700 !important; }
    [data-testid="stLinkButton"] a:hover, a[data-testid="stLinkButton"]:hover { background: #214d35 !important; border-color: #214d35 !important; color: #ffffff !important; }
    [data-testid="stLinkButton"] a *, a[data-testid="stLinkButton"] * { color: #ffffff !important; }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { background: var(--forest) !important; border-color: var(--forest) !important; transform: translateY(-1px); }
    div.stButton > button:focus, div.stFormSubmitButton > button:focus { outline: 3px solid #a8d5a9 !important; outline-offset: 2px; }

    /* All inputs: force high-contrast light controls even if Streamlit/browser uses dark theme */
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextArea"] [data-baseweb="textarea"],
    [data-testid="stDateInput"] [data-baseweb="input"],
    [data-testid="stTimeInput"] [data-baseweb="input"] {
      background: #ffffff !important; border: 1px solid #b8cdb7 !important; border-radius: 10px !important;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input {
      background: #ffffff !important; color: #24372b !important; -webkit-text-fill-color: #24372b !important;
      caret-color: #24372b !important; opacity: 1 !important;
    }
    [data-testid="stNumberInput"] button { background: #edf5e9 !important; color: #214d35 !important; border-left: 1px solid #d7e4d4 !important; }
    [data-testid="stNumberInput"] button svg { fill: #214d35 !important; color: #214d35 !important; }
    [data-baseweb="select"] > div { background: #ffffff !important; border-color: #b8cdb7 !important; }
    [data-baseweb="select"] *, [data-baseweb="popover"] *, [role="listbox"] * { color: #24372b !important; }
    [data-testid="stFileUploader"] section {
      background: #ffffff !important; background-color: #ffffff !important;
      border: 1px dashed #a9c6a5 !important; border-radius: 13px !important;
    }
    [data-testid="stFileUploader"] section * { color: #24372b !important; }
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploader"] section button * {
      background: #3e8051 !important; background-color: #3e8051 !important;
      color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
      border: 1px solid #3e8051 !important; border-radius: 9px !important;
    }
    [data-testid="stFileUploader"] section button:hover,
    [data-testid="stFileUploader"] section button:hover * {
      background: #214d35 !important; background-color: #214d35 !important;
      color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
      border-color: #214d35 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:not(:has(input:checked)):not(:has([role="radio"][aria-checked="true"])) * {
      color: #173b29 !important;
    }

    /* Metrics, alerts, containers */
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--border); border-radius: 15px; padding: 1rem; box-shadow: 0 3px 12px rgba(36,55,43,.035); }
    [data-testid="stMetricLabel"] * { color: #5c7062 !important; }
    [data-testid="stMetricValue"] * { color: var(--forest) !important; }
    [data-testid="stAlert"] { border-radius: 12px; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 15px !important; border-color: var(--border) !important; background: #ffffff; }
    a { color: #28683e !important; }
    a:hover { color: #173b29 !important; }
    hr { border-color: #dce7d7 !important; }

    @media (max-width: 760px) {
      [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 2rem; }
      .hero { padding: 1.4rem; border-radius: 18px; }
      .hero:after { right: .5rem; font-size: 4.5rem; }
      .feature-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Home",
    "Crop Recommendation",
    "Weather Updates",
    "Plant Disease Detection",
    "Government Schemes",
    "About",
]
PAGE_ICONS = {
    "Home": "🏡",
    "Crop Recommendation": "🌾",
    "Weather Updates": "🌦️",
    "Plant Disease Detection": "🍃",
    "Government Schemes": "🏛️",
    "About": "ℹ️",
}


def navigate_to(page_name: str) -> None:
    """Queue navigation safely so Streamlit's radio widget state is not changed after creation."""
    st.session_state["next_page"] = page_name


if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"
if "next_page" in st.session_state:
    requested_page = st.session_state.pop("next_page")
    if requested_page in PAGES:
        st.session_state["selected_page"] = requested_page

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div style="font-size:2.2rem; margin-bottom:.35rem;">🌱</div>
          <div class="brand-name">RaithuMithra AI</div>
          <div class="brand-tagline">A smarter companion for every farmer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**EXPLORE TOOLS**")
    page = st.radio(
        "Navigate",
        PAGES,
        key="selected_page",
        format_func=lambda item: f"{PAGE_ICONS[item]}  {item}",
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="sidebar-foot">
          <div style="font-size:1.3rem; margin-bottom:.25rem;">🌿</div>
          <strong>Grow with confidence.</strong><br>
          Simple digital tools for better-informed farming decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, description: str, icon: str = "🌿") -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(f"{icon} {title}")
    st.markdown(f'<div class="page-intro">{html.escape(description)}</div>', unsafe_allow_html=True)


def load_crop_model():
    if not MODEL_PATH.exists():
        return None, f"Model file not found: `{MODEL_PATH}`"
    try:
        import joblib
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f"Could not load the crop model: {exc}"


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-pill">🌱 YOUR DIGITAL FARMING COMPANION</div>
          <h1>Good decisions start<br>with healthy soil.</h1>
          <p>Welcome to RaithuMithra AI — practical tools to explore crop suggestions, local weather, plant-image guidance, and official farmer schemes in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-heading"><div class="eyebrow">TOOLS FOR YOUR FARM</div><h2>What would you like to do today?</h2></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown('<div class="feature-card"><div class="feature-icon">🌾</div><h3>Crop Recommendation</h3><p>Explore a crop suggestion based on soil nutrients and climate values.</p></div>', unsafe_allow_html=True)
        if st.button("Explore crop tool →", key="home_crop", use_container_width=True):
            navigate_to("Crop Recommendation")
            st.rerun()
    with c2:
        st.markdown('<div class="feature-card"><div class="feature-icon">🌦️</div><h3>Weather Updates</h3><p>Look up current weather conditions for a village, town, or city.</p></div>', unsafe_allow_html=True)
        if st.button("Check local weather →", key="home_weather", use_container_width=True):
            navigate_to("Weather Updates")
            st.rerun()
    with c3:
        st.markdown('<div class="feature-card"><div class="feature-icon">🍃</div><h3>Plant Image Guidance</h3><p>Upload a clear plant or leaf photo to preview it for further review.</p></div>', unsafe_allow_html=True)
        if st.button("Upload a plant image →", key="home_disease", use_container_width=True):
            navigate_to("Plant Disease Detection")
            st.rerun()

    st.markdown(
        '<div class="home-bottom-heading"><div class="eyebrow">MORE SUPPORT</div><h2>Helpful resources for your farm</h2></div>',
        unsafe_allow_html=True,
    )

    info_col, schemes_col = st.columns(2, gap="large")
    with info_col:
        st.markdown(
            """
            <div class="home-info-card">
              <div class="home-card-icon">🌱</div>
              <h3>Smart Farming, Better Decisions</h3>
              <p>Use soil and climate information to explore crop suggestions, check local weather, and make more informed farming decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with schemes_col:
        st.markdown(
            """
            <div class="home-info-card">
              <div class="home-card-icon">🏛️</div>
              <h3>Government Schemes</h3>
              <p>Explore official resources for farmer support, crop insurance, soil health, and agricultural markets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Browse government schemes →", key="home_schemes", use_container_width=True):
            navigate_to("Government Schemes")
            st.rerun()

    st.markdown(
        '<div class="home-note"><strong>💡 Helpful note</strong><br>Use reliable soil-test values and local agricultural advice before making major farming decisions. Scheme eligibility and application dates can change, so check official websites for the latest details.</div>',
        unsafe_allow_html=True,
    )


def render_crop_recommendation() -> None:
    page_header("SOIL + CLIMATE", "Crop Recommendation", "Enter your soil and climate values to get a suggestion from the trained machine-learning model.", "🌾")
    model, model_error = load_crop_model()
    if model is None:
        st.error("The crop recommendation model is not ready.")
        st.write(model_error)
        st.markdown("Make sure the trained model exists at `models/crop_model.joblib`.")
        st.code("python train_crop_model.py", language="powershell")
        return

    st.markdown('<div class="soft-panel"><strong>💡 Tip:</strong> If available, use values from a recent soil test and local weather information.</div>', unsafe_allow_html=True)
    st.markdown("### Soil and climate details")
    with st.form("crop_form"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
            phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
            potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
        with col2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
        submitted = st.form_submit_button("🌱  Recommend a crop", use_container_width=True)

    if submitted:
        try:
            input_data = pd.DataFrame([{
                "n": nitrogen,
                "p": phosphorus,
                "k": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }])
            prediction = model.predict(input_data)
            crop_name = str(prediction[0]).replace("_", " ").title()
            st.markdown("### Your model suggestion")
            st.success(f"Suggested crop: {crop_name} 🌱")
            st.caption("This is a model-based suggestion, not a guarantee. Confirm suitability with local agricultural experts before planting.")
            with st.expander("Review the values you entered"):
                st.dataframe(input_data.rename(columns={"n": "Nitrogen", "p": "Phosphorus", "k": "Potassium", "temperature": "Temperature (°C)", "humidity": "Humidity (%)", "ph": "Soil pH", "rainfall": "Rainfall (mm)"}), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info("Check that the model was trained using these seven columns: n, p, k, temperature, humidity, ph, rainfall.")


def render_weather() -> None:
    page_header("LOCAL CONDITIONS", "Weather Updates", "Check current weather information for a village, town, or city.", "🌦️")
    with st.container(border=True):
        location = st.text_input("Location", placeholder="e.g., Shadnagar, Telangana", help="Enter a place name in English.")
        get_weather = st.button("🌤️  Get current weather", key="get_weather", use_container_width=True)
    if get_weather:
        if not location.strip():
            st.warning("Please enter a location first.")
            return
        try:
            import requests
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location.strip(), "count": 1, "language": "en", "format": "json"},
                timeout=15,
            )
            geo_response.raise_for_status()
            results = geo_response.json().get("results") or []
            if not results:
                st.error("Location not found. Try a nearby town or enter the name in English.")
                return
            place = results[0]
            latitude, longitude = place["latitude"], place["longitude"]
            display_location = ", ".join(str(place.get(k)) for k in ("name", "admin1", "country") if place.get(k))
            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
                timeout=15,
            )
            weather_response.raise_for_status()
            current = weather_response.json().get("current")
            if not current:
                st.error("Weather data is not available for this location right now.")
                return
            st.markdown(f"### 📍 {html.escape(display_location)}")
            a, b, c = st.columns(3, gap="medium")
            a.metric("Temperature", f"{current.get('temperature_2m', 'N/A')} °C")
            b.metric("Feels like", f"{current.get('apparent_temperature', 'N/A')} °C")
            c.metric("Humidity", f"{current.get('relative_humidity_2m', 'N/A')}%")
            d, e = st.columns(2, gap="medium")
            d.metric("Wind speed", f"{current.get('wind_speed_10m', 'N/A')} km/h")
            e.metric("Precipitation", f"{current.get('precipitation', 'N/A')} mm")
            st.caption(f"Updated: {current.get('time', 'time not provided')} (local time for selected location). Weather source: Open-Meteo.")
        except Exception as exc:
            st.error("Unable to retrieve weather information. Check your internet connection and try again.")
            st.caption(f"Technical detail: {exc}")


@st.cache_resource
def load_leaf_health_model():
    """Load the trained leaf-health model once and reuse it across reruns."""
    import tensorflow as tf
    if not LEAF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Leaf-health model not found: {LEAF_MODEL_PATH}")
    return tf.keras.models.load_model(LEAF_MODEL_PATH)


def render_disease_detection() -> None:
    # Keep the original page layout and styling; connect the model to its existing upload flow.
    page_header("PLANT HEALTH", "Plant Image Guidance", "Upload a clear plant-leaf image for preview and further review.", "🍃")
    st.markdown('<div class="soft-panel"><strong>📷 Image upload</strong><br>Upload a clear leaf photo to receive an experimental AI screening result. This prototype does not identify the exact disease.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a plant image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        try:
            import numpy as np
            from PIL import Image

            image = Image.open(uploaded_file)
            left, right = st.columns([1.2, 1], gap="large")
            with left:
                st.image(image, caption="Uploaded plant image", use_container_width=True)
            with right:
                st.markdown("### Image details")
                st.write(f"**File:** {html.escape(uploaded_file.name)}")
                st.write(f"**Dimensions:** {image.width} × {image.height} pixels")
                st.write(f"**Format:** {html.escape(str(image.format or 'Unknown'))}")
                with st.spinner("Checking the leaf image..."):
                    model = load_leaf_health_model()
                    prepared_image = image.convert("RGB").resize((224, 224))
                    image_array = np.asarray(prepared_image, dtype=np.float32)
                    image_array = np.expand_dims(image_array, axis=0)
                    prediction = float(np.asarray(model.predict(image_array, verbose=0)).reshape(-1)[0])

                if prediction >= 0.5:
                    st.warning("**Result: Possibly diseased or damaged**")
                    st.write(f"Model score: {prediction * 100:.1f}%")
                else:
                    st.success("**Result: Possibly healthy**")
                    st.write(f"Model score for possible disease/damage: {prediction * 100:.1f}%")
                st.caption("This is an experimental screening result, not a confirmed diagnosis. The model was trained on a limited dataset and may be inaccurate, especially for unfamiliar plants. Consult a local agriculture expert before treatment.")
        except Exception as exc:
            st.error(f"Could not analyse this image: {exc}")
            st.info("Check that models/leaf_health_model.keras exists and TensorFlow, NumPy, and Pillow are installed.")


def render_schemes() -> None:
    page_header("OFFICIAL RESOURCES", "Government Schemes", "Explore official resources for farmer support, crop insurance, soil health, and agricultural markets.", "🏛️")
    st.info("Scheme rules, eligibility, and application windows can change. Confirm current details on the official website.")
    schemes = [
        ("PM-KISAN", "💰", "Income support scheme information and beneficiary services.", "https://pmkisan.gov.in/", "Income support"),
        ("Pradhan Mantri Fasal Bima Yojana", "🛡️", "Crop insurance information and farmer services.", "https://pmfby.gov.in/", "Crop insurance"),
        ("Soil Health Card", "🌱", "Soil testing and soil health card information.", "https://soilhealth.dac.gov.in/", "Soil testing"),
        ("National Agriculture Market (e-NAM)", "🧺", "Online agricultural market information.", "https://www.enam.gov.in/", "Market access"),
        ("Agriculture Department of Telangana", "🌾", "State agriculture department information and updates.", "https://agri.telangana.gov.in/", "State resources"),
    ]
    for start in range(0, len(schemes), 2):
        cols = st.columns(2, gap="large")
        for col, item in zip(cols, schemes[start:start + 2]):
            title, icon, description, url, category = item
            with col:
                with st.container(border=True):
                    st.markdown(f"<div style='font-size:2rem'>{icon}</div><div class='eyebrow'>{html.escape(category)}</div>", unsafe_allow_html=True)
                    st.subheader(title)
                    st.write(description)
                    st.link_button("Visit official website ↗", url, use_container_width=True)


def render_about() -> None:
    page_header("ABOUT THE PROJECT", "About RaithuMithra AI", "A simple digital companion bringing useful farming tools together in one application.", "🌿")
    st.markdown(
        """
        <div class="hero" style="padding:1.5rem 1.7rem;">
          <h2 style="margin-top:0;">Technology that supports informed farming</h2>
          <p>RaithuMithra AI is designed to make selected digital resources easier to access, with a simple interface and practical guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### What is included")
    items = [
        ("🌾", "Crop recommendation", "A locally stored machine-learning model suggests a crop from seven soil and climate inputs."),
        ("🌦️", "Weather updates", "Current weather information retrieved through Open-Meteo when internet access is available."),
        ("🍃", "Plant image upload", "Upload and preview plant images. Disease classification is not connected yet."),
        ("🏛️", "Government schemes", "Links to official agricultural scheme and service websites."),
    ]
    for icon, title, description in items:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.write(description)
    st.warning("This is an educational support tool, not a replacement for professional agricultural advice. Verify important crop, disease, weather, and scheme decisions with trusted local sources.")


# ----------------------------- Page routing -------------------------------
if page == "Home":
    render_home()
elif page == "Crop Recommendation":
    render_crop_recommendation()
elif page == "Weather Updates":
    render_weather()
elif page == "Plant Disease Detection":
    render_disease_detection()
elif page == "Government Schemes":
    render_schemes()
elif page == "About":
    render_about()
