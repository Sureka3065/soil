import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Soil Health Assessment", layout="wide")

st.title("🌱 Soil Health Assessment & Fertility Prediction App")
st.write("Upload your soil dataset and get fertility predictions with visualizations.")

# ---------------------------- STEP 2: UPLOAD DATASET ----------------------------
uploaded_file = st.file_uploader("Upload Excel/CSV dataset", type=["xlsx", "xls", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Preview of Dataset")
    st.dataframe(df)

    # ---------------------------- STEP 3: SELECT FEATURES & TARGET ----------------------------
    st.subheader("🔧 Select Target Column (Output)")
    target = st.selectbox("Target variable", df.columns)

    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    # ---------------------------- STEP 4: TRAIN / TEST SPLIT ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------- STEP 5: SCALING ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------- STEP 6 & 7: ML MODELS ----------------------------
    st.subheader("🤖 Training Models…")

    model_extra = ExtraTreesRegressor(n_estimators=300, random_state=42)
    model_extra.fit(X_train_scaled, y_train)

    model_cat = CatBoostRegressor(verbose=0)
    model_cat.fit(X_train, y_train)

    st.success("Models Trained Successfully!")

    # ---------------------------- STEP 8: USER INPUT FORM ----------------------------
    st.subheader("📝 Enter Soil Parameters for Prediction")

    input_data = {}
    for col in features:
        input_data[col] = st.number_input(
            f"Enter {col}", value=float(df[col].mean())
        )

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # ---------------------------- STEP 9: PREDICT ----------------------------
    if st.button("🔮 Predict"):
        pred1 = model_extra.predict(input_scaled)[0]
        pred2 = model_cat.predict(input_df)[0]

        st.success("Prediction Completed!")

        st.write(f"🌲 *Extra Trees Prediction:* {pred1}")
        st.write(f"🐈 *CatBoost Prediction:* {pred2}")

        # -------------------------------------------------------------
        # SOIL FERTILITY SCORE FUNCTION
        # -------------------------------------------------------------
        def calculate_soil_fertility_score(row):
            weights = {
                "pH": 0.15,
                "EC": 0.05,
                "Organic_Carbon": 0.20,
                "Nitrogen": 0.15,
                "Phosphorus": 0.15,
                "Potassium": 0.15,
                "Sulphur": 0.05,
                "Zinc": 0.05,
                "Iron": 0.03,
                "Manganese": 0.02
            }

            score = 0
            for col, w in weights.items():
                if col in row:
                    max_val = df[col].max() + 1e-6
                    normalized = max(0, min(1, row[col] / max_val))
                    score += normalized * w

            return round(score * 100, 2)

        # -------------------------------------------------------------
        # STEP 10: FERTILITY SCORE VISUALIZATION
        # -------------------------------------------------------------
        fertility_score = calculate_soil_fertility_score(input_data)

        st.subheader("🌱 Soil Fertility Score")

        if fertility_score < 40:
            category = "Low Fertility"
            color = "red"
        elif fertility_score < 70:
            category = "Medium Fertility"
            color = "orange"
        else:
            category = "High Fertility"
            color = "green"

        st.markdown(f"""
        <div style='padding:15px;border-radius:10px;background-color:{color};color:white;
        font-size:22px;text-align:center;'>
            <b>Soil Fertility Score: {fertility_score}/100</b><br>
            <b>Status: {category}</b>
        </div>
        """, unsafe_allow_html=True)

        st.write("### 🌡 Fertility Gauge")
        st.progress(fertility_score / 100)

        # -------------------------------------------------------------
        # STEP 11: BAR GRAPH (Nutrient Values)
        # -------------------------------------------------------------
        st.write("### 🔬 Nutrient Contribution Bar Graph")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(input_data.keys(), input_data.values())
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
