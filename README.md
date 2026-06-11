# movie-rating-predictor-ml-project-codsoft

# 🎬 Movie Rating Predictor

Predict IMDb ratings for Indian movies using Machine Learning.


---

## 🚀 Live Demo

🔗 [Open Live Demo](https://movie-rating-predictor-ml-project-codsoft-9sxcwzafhiz8xkuddtnf.streamlit.app/)

---
## 📓 Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshaya-Vasireddi/movie-rating-predictor-ml-project-codsoft/blob/main/movie_rating_prediction.ipynb)

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
## 📊 Some Project Previews
<img width="1218" height="403" alt="Screenshot from 2026-06-11 13-40-00" src="https://github.com/user-attachments/assets/3dbed707-4d61-4bbd-a7a6-ca71ccbec8d4" />

<img width="1218" height="450" alt="Screenshot from 2026-06-11 13-40-42" src="https://github.com/user-attachments/assets/648f0d38-0d63-4c18-92d2-107ed701c009" />

<img width="1218" height="450" alt="Screenshot from 2026-06-11 13-41-11" src="https://github.com/user-attachments/assets/9a98855d-17c8-4f80-86e2-a3d2d18c94a7" />

<img width="1218" height="450" alt="Screenshot from 2026-06-11 13-41-19" src="https://github.com/user-attachments/assets/08a75119-906e-49ab-aa53-115c0918ff4a" />

<img width="1218" height="450" alt="Screenshot from 2026-06-11 13-41-35" src="https://github.com/user-attachments/assets/95b1a301-649b-40cf-aedb-cbbc1108c2af" />

<img width="1182" height="550" alt="newplot(1)" src="https://github.com/user-attachments/assets/5c2f3931-9a3f-4fcc-8bc5-abbc9b49dfde" />

<img width="1182" height="500" alt="newplot(2)" src="https://github.com/user-attachments/assets/f769e6ee-297b-48a3-a676-902c6fd4ce53" />



## 👩‍💻 Author

**Akshaya Vasireddi** 
