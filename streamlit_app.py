# app.py
# Line numbers (for reference): 
# 1: import streamlit as st
# 2: import pandas as pd
# 3: import numpy as np
# 4: from sklearn.model_selection import train_test_split
# 5: from sklearn.preprocessing import StandardScaler
# 6: from sklearn.ensemble import ExtraTreesRegressor
# 7: import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

# try importing catboost, but continue if not available
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception as e:
    CATBOOST_AVAILABLE = False
    catboost_import_error = str(e)

st.set_page_config(page_title="Soil Health Assessment", layout="wide")
st.title("🌱 Soil Health Assessment & Fertility Prediction App")

st.write("Upload your soil dataset and get fertility predictions with visualizations.")

# Show helpful message if catboost import failed
if not CATBOOST_AVAILABLE:
    st.warning(
        "CatBoost is not available in this environment. "
        "The app will run Extra Trees only. "
        "To enable CatBoost install: pip install catboost"
    )

# ---------------------------- UPLOAD DATASET ----------------------------
uploaded_file = st.file_uploader("Upload Excel/CSV dataset", type=["xlsx", "xls", "csv"])

if not uploaded_file:
    st.info("Please upload a dataset (.csv or .xlsx). If you already did and see an import error, ensure required packages are installed.")
else:
    # load dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read the uploaded file: {e}")
        st.stop()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # check for target column
    st.subheader("🔧 Select Target Column (Output)")
    target = st.selectbox("Target variable", df.columns)

    # validate that target exists and features numeric where expected
    if target is None:
        st.error("Please select a target column.")
        st.stop()

    features = [c for c in df.columns if c != target]

    # Basic missing value handling
    # Separate numeric and categorical columns (simple heuristic)
    num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    X = df[features].copy()
    y = df[target].copy()

    # Convert categorical features to numeric via one-hot encoding for ExtraTrees
    if len(cat_cols) > 0:
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    else:
        X_encoded = X.copy()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Scaling numeric columns (ExtraTrees doesn't strictly need it but CatBoost might)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.subheader("🤖 Train Models")

    # Train Extra Trees
    try:
        model_extra = ExtraTreesRegressor(n_estimators=200, random_state=42)
        model_extra.fit(X_train_scaled, y_train)
        st.success("✅ Extra Trees trained successfully")
    except Exception as e:
        st.error(f"Extra Trees training failed: {e}")
        st.stop()

    # Train CatBoost if available (use original X with categorical columns handled by CatBoost if present)
    model_cat = None
    if CATBOOST_AVAILABLE:
        try:
            # For CatBoost, use original X (with categorical columns indices)
            X_cb = X.copy()
            # CatBoost requires categorical feature indices when passing DataFrame
            cat_feature_indices = [X_cb.columns.get_loc(c) for c in cat_cols] if cat_cols else []
            model_cat = CatBoostRegressor(verbose=0)
            model_cat.fit(X_cb, y)
            st.success("✅ CatBoost trained successfully")
        except Exception as e:
            st.warning(f"CatBoost training failed: {e}")
            model_cat = None

    # Input form for single prediction
    st.subheader("📝 Enter Soil Parameters for Prediction")

    input_values = {}
    for col in features:
        if col in num_cols:
            val = st.number_input(f"{col}", value=float(df[col].median()))
            input_values[col] = val
        else:
            val = st.text_input(f"{col}", value=str(df[col].mode().iloc[0]) if not df[col].mode().empty else "Unknown")
            input_values[col] = val

    if st.button("🔮 Predict"):
        # Prepare input for ExtraTrees (use same encoding / scaling)
        input_df = pd.DataFrame([input_values])
        if len(cat_cols) > 0:
            input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
            # ensure same columns as training
            for c in X_encoded.columns:
                if c not in input_encoded.columns:
                    input_encoded[c] = 0
            input_encoded = input_encoded[X_encoded.columns]
        else:
            input_encoded = input_df.copy()

        input_scaled = scaler.transform(input_encoded)

        # Extra Trees prediction
        try:
            pred_extra = model_extra.predict(input_scaled)[0]
            st.write(f"🌲 *Extra Trees Prediction:* {pred_extra}")
        except Exception as e:
            st.error(f"Prediction with Extra Trees failed: {e}")

        # CatBoost prediction (if available)
        if model_cat is not None:
            try:
                pred_cat = model_cat.predict(input_df)[0]
                st.write(f"🐈 *CatBoost Prediction:* {pred_cat}")
            except Exception as e:
                st.warning(f"CatBoost prediction failed: {e}")

        # Fertility score calculation
        def calculate_soil_fertility_score(row, df_ref):
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
            score = 0.0
            for col, w in weights.items():
                if col in row and col in df_ref.columns:
                    max_val = df_ref[col].max() + 1e-6
                    try:
                        val = float(row[col])
                        normalized = max(0.0, min(1.0, val / max_val))
                        score += normalized * w
                    except:
                        pass
            return round(score * 100, 2)

        fertility_score = calculate_soil_fertility_score(input_values, df)

        # Visualization
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
        font-size:20px;text-align:center;'>
            <b>Soil Fertility Score: {fertility_score}/100</b><br>
            <b>Status: {category}</b>
        </div>
        """, unsafe_allow_html=True)

        st.write("### 🔬 Nutrient Bar Graph")
        fig, ax = plt.subplots(figsize=(8,4))
        # Only plot numeric inputs for the bar chart
        numeric_input = {k: v for k, v in input_values.items() if k in num_cols}
        ax.bar(numeric_input.keys(), numeric_input.values())
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
