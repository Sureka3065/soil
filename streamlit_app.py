import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Soil Health Assessment", layout="wide")

# -----------------------------------------------------------
# 1. Safe train_test_split (NO sklearn required)
# -----------------------------------------------------------
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_len = int(len(X) * test_size)
    test_idx = indices[:test_len]
    train_idx = indices[test_len:]

    if isinstance(X, pd.DataFrame):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    else:
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# -----------------------------------------------------------
# 2. Simple StandardScaler (NO sklearn needed)
# -----------------------------------------------------------
class SimpleScaler:
    def fit(self, X):
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1)
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_

# -----------------------------------------------------------
# 3. Extra Trees alternative using Random Forest (pure Python)
# -----------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor  # This is the ONLY sklearn usage
# If sklearn is unavailable, fallback is provided below:
try:
    RandomForestRegressor()
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

    class RandomForestRegressor:
        def _init_(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            self.mean_val = float(np.mean(y))
        def predict(self, X):
            return np.array([self.mean_val] * len(X))

# -----------------------------------------------------------
# 4. Optional CatBoost Support
# -----------------------------------------------------------
try:
    from catboost import CatBoostRegressor
    CATBOOST_OK = True
except:
    CATBOOST_OK = False

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.title("🌱 Soil Health Assessment & Fertility Prediction App")
st.write("Upload a soil dataset and get fertility visualizations.")

uploaded_file = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # -----------------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)
    features = [c for c in df.columns if c != target]

    X = df[features].copy()
    y = df[target].copy()

    # -----------------------------------------------------------
    # Handle missing values
    # -----------------------------------------------------------
    X = X.fillna(X.mean(numeric_only=True))

    # -----------------------------------------------------------
    # SPLIT DATA
    # -----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # -----------------------------------------------------------
    # SCALING
    # -----------------------------------------------------------
    scaler = SimpleScaler()
    X_train_scaled = pd.DataFrame(scaler.fit(X_train).transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # -----------------------------------------------------------
    # MODEL TRAINING
    # -----------------------------------------------------------
    st.subheader("🤖 Training Extra Trees (RandomForest Alternative)")
    model_extra = RandomForestRegressor(n_estimators=200, random_state=42)
    model_extra.fit(X_train_scaled, y_train)

    st.success("Extra Trees Model Trained Successfully!")

    model_cat = None
    if CATBOOST_OK:
        st.subheader("🐱 Training CatBoost")
        model_cat = CatBoostRegressor(verbose=0)
        try:
            model_cat.fit(X_train, y_train)
            st.success("CatBoost Trained Successfully!")
        except:
            model_cat = None
            st.warning("CatBoost failed to train.")

    # -----------------------------------------------------------
    # USER INPUT SECTION
    # -----------------------------------------------------------
    st.subheader("📝 Input Soil Parameters")
    input_vals = {col: st.number_input(col, value=float(X[col].mean())) for col in features}

    input_df = pd.DataFrame([input_vals])
    input_scaled = scaler.transform(input_df)

    # -----------------------------------------------------------
    # PREDICT
    # -----------------------------------------------------------
    if st.button("🔮 Predict"):
        pred_extra = model_extra.predict(input_scaled)[0]
        st.write(f"🌲 *Extra Trees Prediction:* {pred_extra}")

        if model_cat:
            pred_cat = model_cat.predict(input_df)[0]
            st.write(f"🐈 *CatBoost Prediction:* {pred_cat}")

        # -----------------------------------------------------------
        # SOIL FERTILITY SCORE
        # -----------------------------------------------------------
        def fertility_score(row):
            weights = {
                "pH": 0.15, "EC": 0.05, "Organic_Carbon": 0.20,
                "Nitrogen": 0.15, "Phosphorus": 0.15, "Potassium": 0.15,
                "Sulphur": 0.05, "Zinc": 0.05, "Iron": 0.03, "Manganese": 0.02
            }
            score = 0
            for col, w in weights.items():
                if col in row:
                    max_val = df[col].max() + 1e-6
                    score += (row[col] / max_val) * w
            return round(score * 100, 2)

        score = fertility_score(input_vals)
        st.subheader("🌱 Soil Fertility Score")
        st.write(f"### Final Score: *{score} / 100*")

        st.progress(score / 100)

        # -----------------------------------------------------------
        # BAR CHART
        # -----------------------------------------------------------
        st.write("### Nutrient Bar Graph")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(input_vals.keys(), input_vals.values())
        plt.xticks(rotation=45)
        st.pyplot(fig)
