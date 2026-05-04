# 🏠 House Price Prediction — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> Predict California housing prices using machine learning. Built an end-to-end pipeline from EDA to a live Streamlit web app.

---

## 🔗 Live Demo
👉 **[Try the App Here](https://your-app-link.streamlit.app)** ← *(update after deploying)*

---

## 📌 Problem Statement
Given features of a California housing block group (income, age, location, etc.), predict the **median house value** of that block.

---

## 📊 Dataset
- **Source:** California Housing Dataset (built into Scikit-learn)
- **Size:** 20,640 samples × 8 features
- **Target:** Median House Value (in $100,000s)

| Feature | Description |
|---|---|
| MedInc | Median income of block group |
| HouseAge | Median age of houses |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block population |
| AveOccup | Average household members |
| Latitude | Block latitude |
| Longitude | Block longitude |

---

## 🔍 Key Findings (EDA)
- **Median Income** is the strongest predictor of house price (correlation: ~0.69)
- Coastal areas (Los Angeles, San Francisco) have significantly higher prices
- House age has minimal impact on price
- Price is capped at $500,000 in this dataset

---

## 🤖 Models Trained & Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **Random Forest** ✅ | **~0.805** | **~$53,000** | **~$37,000** |
| Gradient Boosting | ~0.777 | ~$57,000 | ~$41,000 |
| Decision Tree | ~0.622 | ~$75,000 | ~$50,000 |
| Linear Regression | ~0.576 | ~$80,000 | ~$58,000 |

> ✅ **Best Model: Random Forest** — explains ~80.5% of variance in house prices

---

## 📁 Project Structure

```
house-price-prediction/
├── house_price_prediction.ipynb  ← Full analysis notebook
├── app.py                        ← Streamlit web app
├── requirements.txt              ← Dependencies
├── model/
│   ├── house_price_model.pkl     ← Saved Random Forest model
│   └── scaler.pkl                ← Saved StandardScaler
└── plots/
    ├── 01_price_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_income_vs_price.png
    ├── 04_geo_distribution.png
    ├── 05_model_comparison.png
    ├── 06_actual_vs_predicted.png
    └── 07_feature_importance.png
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/siddharth0444/house-price-prediction.git
cd house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Jupyter notebook (builds the model)
jupyter notebook house_price_prediction.ipynb

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **ML:** Scikit-learn
- **Web App:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 📸 Screenshots

*(Add screenshots of your notebook plots and app here after running)*

---

## 👤 Author

**Siddharth Rajak**
- GitHub: [@siddharth0444](https://github.com/siddharth0444)
- LinkedIn: [in/siddharthrajak](https://linkedin.com/in/siddharthrajak)
- Email: devrajak22@gmail.com

---

*⭐ If you found this useful, give it a star!*

Screenshot 2026-05-04 at 6.59.49 PMScreenshot 2026-05-04 at 6.59.46 PMScreenshot 2026-05-04 at 6.59.38 PMScreenshot 2026-05-04 at 6.59.18 PM
