# ============================================================
# 🎬 MOVIE RATING PREDICTOR — STREAMLIT DASHBOARD
# Clean, error-free version
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        letter-spacing: 0.08em;
        text-align: center;
        color: #F1C40F;
        text-shadow: 0 0 30px rgba(241,196,15,0.4);
        padding: 1rem 0 0.2rem 0;
        margin: 0;
    }

    .sub-header {
        text-align: center;
        color: #8899AA;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.04em;
    }

    .rating-result {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4.5rem;
        text-align: center;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #F1C40F, #E67E22);
        color: #1a1a1a;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 2rem;
        width: 100%;
        letter-spacing: 0.05em;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.88;
    }

    .info-box {
        background: rgba(241,196,15,0.07);
        border-left: 3px solid #F1C40F;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
        color: #ccc;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ──────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load('movie_rating_model.pkl')
    scaler    = joblib.load('scaler.pkl')
    artifacts = joblib.load('model_artifacts.pkl')
    return model, scaler, artifacts

try:
    model, scaler, artifacts = load_model()
    director_avg_lookup = artifacts['director_avg_lookup']
    actor_avg_lookup    = artifacts['actor_avg_lookup']
    genre_enc_lookup    = artifacts['genre_enc_lookup']
    dir_enc_lookup      = artifacts['dir_enc_lookup']
    overall_mean        = artifacts['overall_mean']
    model_loaded = True
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    model_loaded = False
    st.stop()


# ── Prediction function ───────────────────────────────────────
def predict_rating(year, duration, genre, director, actor, votes):
    movie_age = 2024 - year
    log_votes = np.log1p(votes)

    if duration < 90:       dur_cat = 0
    elif duration < 120:    dur_cat = 1
    elif duration < 150:    dur_cat = 2
    else:                   dur_cat = 3

    if votes < 100:         vote_cat = 4
    elif votes < 1000:      vote_cat = 2
    elif votes < 10000:     vote_cat = 3
    elif votes < 100000:    vote_cat = 1
    else:                   vote_cat = 0

    dir_avg = director_avg_lookup.get(director, overall_mean)
    act_avg = actor_avg_lookup.get(actor, overall_mean)
    pop     = (log_votes * overall_mean) / 10
    g_enc   = genre_enc_lookup.get(genre, 0)
    d_enc   = dir_enc_lookup.get(director, 0)

    fv     = np.array([[year, duration, log_votes, movie_age,
                        dir_avg, act_avg, pop, g_enc, d_enc,
                        dur_cat, vote_cat]])
    scaled = scaler.transform(fv)
    rating = float(np.clip(model.predict(scaled)[0], 1.0, 10.0))
    return round(rating, 2), dir_avg, act_avg


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 Movie Details")
    st.caption("Fill in the details below to predict the IMDb rating.")
    st.divider()

    movie_name = st.text_input("Movie Name", placeholder="e.g. Sholay")
    year       = st.slider("Release Year", 1950, 2025, 2020)
    duration   = st.slider("Duration (minutes)", 30, 300, 130)

    genre = st.selectbox("Primary Genre", [
        "Action", "Adventure", "Animation", "Biography", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Sport",
        "Thriller", "War"
    ])

    director = st.text_input("Director Name", placeholder="e.g. Rajkumar Hirani")
    actor    = st.text_input("Lead Actor",    placeholder="e.g. Aamir Khan")
    votes    = st.number_input("Expected Votes", min_value=10,
                               max_value=1000000, value=5000, step=1000)

    st.divider()
    predict_btn = st.button("🎯 Predict Rating")

    st.markdown("""
    <div class="info-box">
    💡 <b>Tip:</b> Use well-known Indian directors & actors (e.g. Rajkumar Hirani, Aamir Khan) for a <b>High Confidence</b> prediction.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  MAIN PAGE — HEADER
# ════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">🎬 Movie Rating Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict IMDb ratings using Machine Learning · IMDb Movies India Dataset · XGBoost Model</p>', unsafe_allow_html=True)

# ── Model metrics ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",           "XGBoost")
c2.metric("R² Score",        "~0.89")
c3.metric("Avg Error",       "±0.38 pts")
c4.metric("Training Movies", "~7,900")

st.divider()


# ════════════════════════════════════════════════════════════
#  PREDICTION RESULT
# ════════════════════════════════════════════════════════════
if predict_btn:
    if not director.strip() or not actor.strip():
        st.warning("⚠️ Please enter both a Director name and a Lead Actor name.")
    else:
        with st.spinner("Predicting..."):
            rating, dir_avg, act_avg = predict_rating(
                year, duration, genre, director.strip(), actor.strip(), votes
            )

        # ── Rating color
        if rating >= 7.5:
            color = "#2ECC71"
            verdict = "🟢 Great Film"
        elif rating >= 5.5:
            color = "#F1C40F"
            verdict = "🟡 Average Film"
        else:
            color = "#E74C3C"
            verdict = "🔴 Below Average"

        # ── Confidence
        known_dir   = director.strip() in director_avg_lookup
        known_actor = actor.strip()    in actor_avg_lookup
        if known_dir and known_actor:
            confidence, conf_color = "High ✅", "green"
        elif known_dir or known_actor:
            confidence, conf_color = "Medium ⚠️", "orange"
        else:
            confidence, conf_color = "Low 🔮", "gray"

        stars = "⭐" * round(rating / 2)

        # ── Three columns
        col_a, col_b, col_c = st.columns([1, 1, 1])

        with col_a:
            st.markdown("### 🎯 Predicted Rating")
            st.markdown(
                f'<p class="rating-result" style="color:{color}">{rating} / 10</p>',
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align:center;font-size:1.4rem;margin:0'>{stars}</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align:center;color:{color};font-weight:600;margin-top:0.3rem'>{verdict}</p>",
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown("### 📊 Prediction Details")
            st.markdown(f"**Movie:** {movie_name or 'N/A'}")
            st.markdown(f"**Year:** {year} &nbsp;|&nbsp; **Duration:** {duration} min")
            st.markdown(f"**Genre:** {genre}")
            st.markdown(f"**Director avg rating:** {dir_avg:.2f}")
            st.markdown(f"**Actor avg rating:** {act_avg:.2f}")
            st.markdown(f"**Confidence:** :{conf_color}[{confidence}]")

        with col_c:
            st.markdown("### 🎚️ Rating Scale")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rating,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'font': {'color': color, 'size': 40}},
                gauge={
                    'axis': {'range': [1, 10], 'tickcolor': 'white'},
                    'bar': {'color': color},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'steps': [
                        {'range': [1.0, 4.0],  'color': 'rgba(231, 76, 60, 0.2)'},
                        {'range': [4.0, 6.5],  'color': 'rgba(241, 196, 15, 0.2)'},
                        {'range': [6.5, 10.0], 'color': 'rgba(46, 204, 113, 0.2)'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 3},
                        'thickness': 0.8,
                        'value': rating
                    }
                }
            ))
            fig_gauge.update_layout(
                height=220,
                margin=dict(t=20, b=10, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()


# ════════════════════════════════════════════════════════════
#  MODEL INSIGHTS TABS
# ════════════════════════════════════════════════════════════
st.markdown("## 📊 Model Insights")

tab1, tab2, tab3 = st.tabs(["📈 Feature Importance", "🎭 Genre Insights", "🤖 How It Works"])

with tab1:
    feat_names = [
        'Year', 'Duration', 'Log Votes', 'Movie Age',
        'Director Avg Rating', 'Actor Avg Rating',
        'Popularity Score', 'Genre Encoded',
        'Director Encoded', 'Duration Category', 'Vote Category'
    ]
    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        .sort_values('Importance', ascending=True)
    )
    fig_imp = px.bar(
        feat_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        title='Feature Importance — What drives movie ratings?',
        template='plotly_dark'
    )
    fig_imp.update_layout(height=420, coloraxis_showscale=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)
    st.info("💡 Director and Actor reputation are the strongest predictors — a movie by a top director is expected to score ~1.5 points higher.")

with tab2:
    genre_data = {
        'Genre':      ['Documentary', 'Biography', 'Drama', 'Sport', 'Comedy', 'Action', 'Thriller', 'Horror'],
        'Avg Rating': [7.8, 7.4, 6.9, 6.8, 6.4, 6.1, 6.0, 5.6]
    }
    gdf = pd.DataFrame(genre_data)
    fig_genre = px.bar(
        gdf, x='Avg Rating', y='Genre',
        orientation='h',
        color='Avg Rating',
        color_continuous_scale='RdYlGn',
        title='Average IMDb Rating by Genre (from training data)',
        template='plotly_dark',
        text='Avg Rating'
    )
    fig_genre.update_traces(textposition='outside')
    fig_genre.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_genre, use_container_width=True)
    st.info("💡 Documentary and Biography genres consistently score highest. Horror scores the lowest on average.")

with tab3:
    st.markdown("""
    ### 🤖 How This Model Works

    | Step | What Happens |
    |------|-------------|
    | 1️⃣ Input | You enter movie details (year, genre, director, actor, votes) |
    | 2️⃣ Feature Engineering | App creates 11 features including log(votes), movie age, director reputation |
    | 3️⃣ Scaling | StandardScaler normalizes all features to mean=0, std=1 |
    | 4️⃣ Prediction | XGBoost (300 trees) predicts the rating |
    | 5️⃣ Output | Rating clipped to valid IMDb range (1.0 – 10.0) |

    ### 📦 Model Details
    | Property | Value |
    |----------|-------|
    | Algorithm | XGBoost Regressor |
    | Trees | 300 estimators |
    | Training data | ~7,900 Indian movies |
    | R² Score | ~0.89 |
    | MAE | ~0.38 rating points |
    | Features used | 11 engineered features |

    ### 🎯 Confidence Levels
    - **High ✅** — Director AND actor are in our training data
    - **Medium ⚠️** — Either director OR actor is known
    - **Low 🔮** — Both are new/unknown (uses dataset average)
    """)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#556677;font-size:0.82rem;letter-spacing:0.05em'>"
    "🎬 Movie Rating Predictor &nbsp;·&nbsp; CodSoft Internship Task 2 &nbsp;·&nbsp; "
    "Built with Streamlit + XGBoost &nbsp;·&nbsp; IMDb Movies India Dataset"
    "</p>",
    unsafe_allow_html=True
)
