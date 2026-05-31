# movie-rating-predictor-ml-project-codsoft

# 🎬 Movie Rating Predictor

Predict IMDb ratings for Indian movies using Machine Learning.


---

## 🚀 Live Demo

🔗 [Open App](https://movie-rating-predictor-ml-project-codsoft-9sxcwzafhiz8xkuddtnf.streamlit.app/)

---

## 📌 About

This app predicts the IMDb rating of an Indian movie based on details like the director, lead actor, genre, release year, duration, and expected votes — using a trained **XGBoost** model.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Regressor |
| Training Data | ~7,900 Indian movies (IMDb India Dataset) |
| R² Score | ~0.89 |
| Mean Absolute Error | ±0.38 rating points |
| Features Used | 11 engineered features |

---

## 🎯 How to Use

1. Open the app using the link above
2. Fill in the movie details in the sidebar:
   - Movie name, release year, duration
   - Genre, director name, lead actor
   - Expected number of votes
3. Click **🎯 Predict Rating**
4. View the predicted IMDb rating, confidence level, and gauge chart

### Example Input
| Field | Value |
|-------|-------|
| Movie Name | 3 Idiots |
| Release Year | 2009 |
| Duration | 170 min |
| Genre | Comedy |
| Director | Rajkumar Hirani |
| Lead Actor | Aamir Khan |
| Expected Votes | 300000 |

---

## 📁 Project Structure

├── app.py                  # Streamlit dashboard
├── movie_rating_model.pkl  # Trained XGBoost model
├── model_artifacts.pkl     # Director/actor/genre lookup tables
├── scaler.pkl              # Fitted StandardScaler
└── requirements.txt        # Python dependencies
---

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Web app framework
- **XGBoost** — ML model
- **Pandas / NumPy** — Data processing
- **Plotly** — Interactive charts
- **Scikit-learn** — Preprocessing

---

## 📊 Key Insight

> Director and Actor reputation are the **strongest predictors** of a movie's rating.
> A film by a top-rated director can score ~1.5 points higher than average.

---

## 👩‍💻 Author

**Akshaya Vasireddi** 
