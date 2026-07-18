import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    /* Hide default header */
    header[data-testid="stHeader"] { background: transparent; }

    /* Hero title */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.15;
    }
    .hero-accent {
        color: #58a6ff;
    }
    .hero-sub {
        font-size: 1rem;
        color: #8b949e;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* Upload zone */
    .upload-card {
        background: #161b22;
        border: 1.5px dashed #30363d;
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
        transition: border-color 0.2s;
    }
    .upload-card:hover { border-color: #58a6ff; }

    /* Result card */
    .result-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.2rem;
    }
    .result-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 0;
    }
    .result-conf {
        font-size: 0.9rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    /* Prob bar label */
    .bar-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-bottom: 2px;
    }

    /* Info badge */
    .badge {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        color: #9ca3af;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 99px;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Section divider */
    .divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.8rem 0;
    }

    /* Note box */
    .note-box {
        background: #1c2128;
        border-left: 3px solid #f78166;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem;
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 1rem;
    }
    .note-box b { color: #e6edf3; }

    /* Model info box */
    .info-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .info-box b { color: #c9d1d9; }

    /* Streamlit file uploader tweak */
    [data-testid="stFileUploader"] > div {
        background: #161b22 !important;
        border: 1.5px dashed #30363d !important;
        border-radius: 12px !important;
    }

    /* Progress bar color */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1f6feb, #58a6ff);
        border-radius: 99px;
    }

    /* Streamlit image caption */
    .stImage > div > div > p { color: #8b949e !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES = {
    0: ("Auto Rickshaw", "🛺"),
    1: ("Bike",          "🚲"),
    2: ("Car",           "🚗"),
    3: ("Motorcycle",    "🏍️"),
    4: ("Plane",         "✈️"),
    5: ("Ship",          "🚢"),
    6: ("Train",         "🚆"),
}
IMG_SIZE = 64

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "vehicle_decision_tree_model.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def preprocess(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.flatten().reshape(1, -1)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 0.5rem 0;">
    <p class="hero-title">Vehicle <span class="hero-accent">Classifier</span></p>
    <p class="hero-sub">Decision Tree Model &nbsp;·&nbsp; 7 Vehicle Types &nbsp;·&nbsp; 64×64 px input</p>
</div>
""", unsafe_allow_html=True)

# Badges
st.markdown("""
<div style="margin-bottom:1rem;">
    <span class="badge">🛺 Auto Rickshaw</span>
    <span class="badge">🚲 Bike</span>
    <span class="badge">🚗 Car</span>
    <span class="badge">🏍️ Motorcycle</span>
    <span class="badge">✈️ Plane</span>
    <span class="badge">🚢 Ship</span>
    <span class="badge">🚆 Train</span>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

if model is None:
    st.markdown("""
    <div class="note-box">
        <b>⚠️ Model file nahi mili!</b><br>
        <code>vehicle_decision_tree_model.pkl</code> is folder mein rakh do jahan <code>app.py</code> hai.<br><br>
        Notebook mein yeh cell run karo:<br>
        <code>import pickle<br>pickle.dump(final_clf, open("vehicle_decision_tree_model.pkl", "wb"))</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown("### 📸 Image Upload Karein")
uploaded = st.file_uploader(
    "Koi bhi vehicle ki photo choose karein",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_res:
        with st.spinner("Analyzing..."):
            arr = preprocess(image)
            pred_label = model.predict(arr)[0]
            name, emoji = CATEGORIES[pred_label]

            has_proba = hasattr(model, "predict_proba")
            if has_proba:
                proba = model.predict_proba(arr)[0]
                confidence = proba[pred_label] * 100
            else:
                confidence = None

        st.markdown(f"""
        <div class="result-card">
            <p style="font-size:3rem;margin:0;">{emoji}</p>
            <p class="result-label">{name}</p>
            {"<p class='result-conf'>Confidence: <b style='color:#58a6ff'>" + f"{confidence:.1f}%" + "</b></p>" if confidence else ""}
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    if has_proba:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Sab categories ki probability:**")
        # Sort by probability descending
        sorted_cats = sorted(CATEGORIES.items(), key=lambda x: proba[x[0]], reverse=True)
        for idx, (cat_name, cat_emoji) in sorted_cats:
            prob = proba[idx]
            bar_color = "#58a6ff" if idx == pred_label else "#30363d"
            label_color = "#e6edf3" if idx == pred_label else "#8b949e"
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="font-size:0.82rem;color:{label_color};margin-bottom:3px;">
                    {cat_emoji} {cat_name} — <b>{prob*100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#484f58;">
        <p style="font-size:3rem;margin:0;">🖼️</p>
        <p style="margin-top:0.5rem;font-size:0.95rem;">Koi image upload karein prediction ke liye</p>
    </div>
    """, unsafe_allow_html=True)

# ── Model Info ────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

with st.expander("ℹ️ Model ke baare mein"):
    depth = getattr(model, 'max_depth', 'N/A')
    features = getattr(model, 'n_features_in_', IMG_SIZE*IMG_SIZE*3)
    st.markdown(f"""
    <div class="info-box">
        <b>Algorithm:</b> Decision Tree Classifier<br>
        <b>Max Depth:</b> {depth}<br>
        <b>Input Features:</b> {features} (= 64 × 64 × 3 pixels)<br>
        <b>Classes:</b> 7 vehicle types<br><br>
        <b>Kaise kaam karta hai?</b><br>
        Image ko 64×64 pe resize kiya jata hai → RGB pixels ko ek 12,288-number ki list mein convert kiya jata hai → Decision Tree pattern match karta hai → result aata hai.
    </div>
    <div class="note-box" style="margin-top:0.8rem;">
        <b>Tip:</b> Decision Tree raw pixels ke saath achha kaam karta hai lekin CNN (Convolutional Neural Network) images ke liye kaafi behtar hota — future improvement!
    </div>
    """, unsafe_allow_html=True)
