# app.py
# Fraud Detection — Streamlit app (XGBoost-only production)
# Place this file in the same folder as "Fraud Detection Transactions Dataset.csv"
# requirements.txt should include:
# streamlit
# pandas
# numpy
# scikit-learn
# matplotlib
# seaborn
# xgboost
# imbalanced-learn
# pickle-mixin

import os
import pickle
import warnings
from math import ceil

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE

# Try import XGBoost (app is XGBoost-only for production)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# baseline models for quick comparison only
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# suppress verbose warnings
warnings.filterwarnings("ignore")

# App config
st.set_page_config(page_title="Fraud Detection — XGBoost", page_icon="💳", layout="wide")
sns.set(style="whitegrid")

# Paths / constants
DATA_PATH = "Fraud Detection Transactions Dataset.csv"
MODEL_PATH = "fraud_xgb_model.pkl"
SCALER_PATH = "fraud_scaler.pkl"
ENCODERS_PATH = "fraud_encoders.pkl"

KEEP_COLS = [
    "Transaction_Amount", "Transaction_Type", "Account_Balance", "Device_Type",
    "Merchant_Category", "Previous_Fraudulent_Activity", "Daily_Transaction_Count",
    "Avg_Transaction_Amount_7d", "Failed_Transaction_Count_7d", "Authentication_Method", "Fraud_Label"
]


# ---------------- Helpers ----------------
@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_df(df: pd.DataFrame, keep_cols=KEEP_COLS):
    """Keep selected columns and label-encode object columns. Return (df_encoded, encoders)."""
    df = df.copy()
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[keep_cols].copy()

    # Normalize Previous_Fraudulent_Activity (No/Yes) -> 0/1 if needed
    if df["Previous_Fraudulent_Activity"].dtype == object:
        df["Previous_Fraudulent_Activity"] = df["Previous_Fraudulent_Activity"].map({"No": 0, "Yes": 1}).fillna(
            df["Previous_Fraudulent_Activity"])
    df["Previous_Fraudulent_Activity"] = df["Previous_Fraudulent_Activity"].astype(int)

    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def get_train_test_scaled_balanced(X, y):
    """Split, scale, and SMOTE-balance training data. Return X_train_bal, y_train_bal, X_test_s, y_test, scaler."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)
    return X_train_bal, y_train_bal, X_test_s, y_test, scaler


def train_xgb(df_encoded, save_artifacts=True, xgb_params=None):
    """
    Train XGBoost on df_encoded using StandardScaler + SMOTE.
    If xgb_params is provided, use them for XGBClassifier init.
    Returns model, scaler, report_str, confusion_matrix, (fpr,tpr,auc)
    """
    X = df_encoded.drop("Fraud_Label", axis=1)
    y = df_encoded["Fraud_Label"]

    X_train_bal, y_train_bal, X_test_s, y_test, scaler = get_train_test_scaled_balanced(X, y)

    params = xgb_params or {}
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1, **params)
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test_s)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    y_probs = model.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    if save_artifacts:
        save_pickle(model, MODEL_PATH)
        save_pickle(scaler, SCALER_PATH)

    return model, scaler, report, cm, (fpr, tpr, roc_auc)


def baseline_comparison(df_encoded):
    """Train basic baseline models for comparison (logistic, knn, dt, randomforest, xg if available). Returns dict results with accuracy."""
    X = df_encoded.drop("Fraud_Label", axis=1)
    y = df_encoded["Fraud_Label"]
    X_train_bal, y_train_bal, X_test_s, y_test, _ = get_train_test_scaled_balanced(X, y)

    models = {
        "Logistic": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train_bal, y_train_bal)
        preds = m.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        results[name] = {"accuracy": acc, "report": classification_report(y_test, preds, output_dict=False)}
    # Add XGBoost baseline if available
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        xgb.fit(X_train_bal, y_train_bal)
        preds = xgb.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        results["XGBoost (baseline)"] = {"accuracy": acc, "report": classification_report(y_test, preds, output_dict=False)}
    return results


# ---------------- Navigation ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Overview", "Dataset", "EDA", "Models & Train (XGBoost)", "Predict"])


# ---------------- Pages ----------------
if page == "Overview":
    st.title("Fraud Transaction prediction")
    st.markdown("""
    #### Problem (expanded)
    Fraudulent transactions cause direct monetary loss and indirect costs such as manual review, customer churn and reputational damage.
    Transaction metadata (amount, merchant category, device, authentication method, previous fraud history, etc.) often contains
    signal that a transaction is anomalous. We build a compact ML pipeline that converts that signal into a probability score to help
    prioritize automated or manual reviews.

    #### Why this approach
    - Keep features small and explainable for fast inference and easy integration.  
    - Use deterministic label-encoding + StandardScaler for reproducible preprocessing.  
    - Handle class imbalance with SMOTE during training so the model learns fraud patterns.  
    - Use **XGBoost as production model** because it captures non-linear interactions, includes regularization and is optimized for speed.  
    - Provide baseline models for transparency and model-selection context.

    #### Objective & Deliverables
    1. Cleaned dataset + reproducible preprocessing.  
    2. EDA with boxplots, distributions, correlation and class-balance checks.  
    3. Baseline model comparison (Logistic, KNN, DecisionTree, RandomForest, XGBoost).   
    4. Interactive Predict UI returning probability (%) and a human-friendly risk label.
    """)
    if not XGBOOST_AVAILABLE:
        st.warning("XGBoost is NOT installed in this environment. Production training/tuning is disabled until you install XGBoost (`pip install xgboost`).")

elif page == "Dataset":
    st.header("Dataset overview")
    df_raw = load_data()
    if df_raw is None:
        st.error(f"Dataset `{DATA_PATH}` not found. Upload it to the app folder.")
        st.stop()

    st.write(f"**Shape:** `{df_raw.shape}`")
    st.subheader("Columns")
    st.write(list(df_raw.columns))
    st.subheader("Preview (first 8 rows)")
    st.dataframe(df_raw.head(8))
    st.subheader("Dtypes & missing counts")
    info_df = pd.DataFrame({"dtype": df_raw.dtypes.astype(str), "missing": df_raw.isnull().sum()})
    st.dataframe(info_df)
    st.markdown("**Dropped / not used columns (project decision):**")
    st.write("- `Transaction_ID`, `User_ID`, `Timestamp`, `IP_Address_Flag`, `Card_Type`, `Card_Age`, `Transaction_Distance`, `Risk_Score`, `Is_Weekend`, `Location`")

elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    df_raw = load_data()
    if df_raw is None:
        st.error("Dataset not found.")
        st.stop()

    eda_cols = [c for c in KEEP_COLS if c in df_raw.columns]
    df_eda = df_raw[eda_cols].copy()

    # Boxplots (show up to all present features, arranged in rows)
    st.subheader("Boxplots (detect outliers)")
    numcols = ["Transaction_Amount", "Account_Balance", "Avg_Transaction_Amount_7d", "Daily_Transaction_Count", "Failed_Transaction_Count_7d"]
    present = [c for c in numcols if c in df_eda.columns]
    if present:
        n = len(present)
        ncols = 3
        nrows = ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = axes.flatten()
        for i, c in enumerate(present):
            sns.boxplot(y=df_eda[c], ax=axes[i], palette="Set3")
            axes[i].set_title(c)
        # hide extra axes
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

    # Distributions (hist + kde) - show all present, arranged similarly
    st.subheader("Distributions (histogram + KDE) by class")
    if present:
        n = len(present)
        ncols = 3
        nrows = ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = axes.flatten()
        for i, c in enumerate(present):
            sns.histplot(data=df_eda, x=c, hue="Fraud_Label", kde=True, stat="density", ax=axes[i], bins=40)
            axes[i].set_title(c)
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

    # Correlation heatmap (reduced size)
    st.subheader("Correlation heatmap (numeric)")
    corr = df_eda.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(6, 4))  # reduced size
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Class distribution
    st.subheader("Class distribution (imbalance)")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="Fraud_Label", data=df_eda, palette="magma", ax=ax)
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    st.pyplot(fig)
    counts = df_eda["Fraud_Label"].value_counts()
    st.write(counts)

elif page == "Models & Train (XGBoost)":
    st.header("Model Selection")
    df_raw = load_data()
    if df_raw is None:
        st.error("Dataset not found.")
        st.stop()

    try:
        df_encoded, encoders = preprocess_df(df_raw, KEEP_COLS)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Preprocessed sample (encoded)")
    st.dataframe(df_encoded.head(6))

    # --------------- Baseline comparison (auto) ----------------
    st.markdown("### Baseline comparison")
    if "baseline_results" not in st.session_state:
        with st.spinner("Running baseline models (Logistic / KNN / DecisionTree / RandomForest / XGBoost"):
            st.session_state["baseline_results"] = baseline_comparison(df_encoded)

    results = st.session_state["baseline_results"]
    acc_df = pd.DataFrame([{"Model": k, "Accuracy": results[k]["accuracy"]} for k in results]).sort_values("Accuracy", ascending=False)
    st.dataframe(acc_df.style.format({"Accuracy": "{:.3f}"}))
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x="Accuracy", y="Model", data=acc_df, palette="viridis", ax=ax)
    ax.set_xlim(0, 1)
    st.pyplot(fig)
    st.markdown("**Baseline reports**")
    for k in results:
        st.markdown(f"**{k}**")
        st.text(results[k]["report"])

    st.markdown("---")

    # --------------- Production model display ----------------
    st.subheader("Production model")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_pickle(MODEL_PATH)
        scaler = load_pickle(SCALER_PATH)
        # Evaluate loaded model on the app's dataset test-split (uses saved scaler to transform)
        X = df_encoded.drop("Fraud_Label", axis=1)
        y = df_encoded["Fraud_Label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        try:
            X_test_s = scaler.transform(X_test)
        except Exception:
            # fallback to creating a new scaler if mismatch; still try to evaluate
            _, _, X_test_s, y_test, _ = get_train_test_scaled_balanced(X, y)
        y_pred = model.predict(X_test_s)
        report = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)
        st.text(report)
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])
        st.pyplot(fig)
        # ROC if available
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            st.subheader("ROC Curve (saved model)")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", lw=2)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("No saved production XGBoost artifact found in app folder.")
        st.write("If you have a trained XGBoost model + scaler + encoders, upload them below so the app can display metrics and the Predict page can use them.")
        uploaded_model = st.file_uploader("Upload tuned model (pickle)", type=["pkl", "pickle"])
        uploaded_scaler = st.file_uploader("Upload scaler (pickle)", type=["pkl", "pickle"])
        uploaded_encoders = st.file_uploader("Upload encoders (pickle)", type=["pkl", "pickle"])
        if uploaded_model and uploaded_scaler and uploaded_encoders:
            save_pickle(pickle.load(uploaded_model), MODEL_PATH)
            save_pickle(pickle.load(uploaded_scaler), SCALER_PATH)
            save_pickle(pickle.load(uploaded_encoders), ENCODERS_PATH)
            st.success("Artifacts uploaded and saved. Refresh or go to Predict page to use the model.")

    st.markdown("### Why choose XGBoost over the baseline models?")
    st.write(
        "- **Better accuracy / F1 on tabular data:** gradient boosting captures non-linear feature interactions that simple linear models miss.  \n"
        "- **Regularization & robustness:** XGBoost provides shrinkage (learning_rate), tree regularization and is less prone to overfitting than naive trees.  \n"
        "- **Speed and production maturity:** optimized implementation with multi-threading and predictable inference latency."
    )

elif page == "Predict":
    st.header("Predict")
    # Ensure artifacts exist
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODERS_PATH)):
        st.warning("Model, scaler or encoders missing. Add artifacts on 'Models & Train (XGBoost)' or upload saved artifacts to the app folder.")
        st.stop()

    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    encoders = load_pickle(ENCODERS_PATH)

    st.subheader("Enter transaction details")
    col1, col2 = st.columns(2)
    with col1:
        txn_amount = st.number_input("Transaction Amount", min_value=0.0, value=50.0, step=1.0)
        txn_type = st.selectbox("Transaction Type", options=list(encoders["Transaction_Type"].classes_))
        account_balance = st.number_input("Account Balance", min_value=0.0, value=10000.0, step=100.0)
        device_type = st.selectbox("Device Type", options=list(encoders["Device_Type"].classes_))
        merchant = st.selectbox("Merchant Category", options=list(encoders["Merchant_Category"].classes_))
    with col2:
        prev_fraud = st.selectbox("Previous Fraudulent Activity", options=["No", "Yes"])
        daily_count = st.number_input("Daily Transaction Count", min_value=0, max_value=1000, value=3)
        avg_7d = st.number_input("Average Transaction Amount (7d)", min_value=0.0, value=120.0)
        failed_7d = st.number_input("Failed Transaction Count (7d)", min_value=0, value=0)
        auth_method = st.selectbox("Authentication Method", options=list(encoders["Authentication_Method"].classes_))

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Transaction_Amount": txn_amount,
            "Transaction_Type": txn_type,
            "Account_Balance": account_balance,
            "Device_Type": device_type,
            "Merchant_Category": merchant,
            "Previous_Fraudulent_Activity": 1 if prev_fraud == "Yes" else 0,
            "Daily_Transaction_Count": daily_count,
            "Avg_Transaction_Amount_7d": avg_7d,
            "Failed_Transaction_Count_7d": failed_7d,
            "Authentication_Method": auth_method
        }])

        # Apply saved encoders
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        # Scale & predict
        X_input_s = scaler.transform(input_df)
        pred = model.predict(X_input_s)[0]
        proba = model.predict_proba(X_input_s)[0][1] if hasattr(model, "predict_proba") else None

        # show input summary
        st.markdown("**Input summary**")
        st.write(input_df)

        st.markdown("### Result")
        if proba is not None:
            pct = float(proba * 100)
            st.markdown(f"<h2 style='text-align:center'>{pct:.2f}% chance of Fraud</h2>", unsafe_allow_html=True)
            st.progress(min(max(int(round(pct)), 0), 100))

            # Visual probability
            probs_df = pd.DataFrame({"Probability": [1 - proba, proba]}, index=["Legit (0)", "Fraud (1)"])
            st.subheader("Class probabilities")
            st.bar_chart(probs_df)

            # Simple risk label:
            if pct >= 70:
                st.error("Risk label: HIGH — immediate review suggested.")
            elif pct >= 40:
                st.warning("Risk label: MEDIUM — review recommended.")
            else:
                st.success("Risk label: LOW — likely legitimate.")
        else:
            # fallback if predict_proba not available
            if pred == 1:
                st.error("🔴 FRAUD (model prediction)")
            else:
                st.success("🟢 LEGIT (model prediction)")

 
# -- end pages --
# No footer text (removed per request)
