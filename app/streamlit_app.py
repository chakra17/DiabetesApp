# app/streamlit_app.py
# Diabetes detection — Streamlit front-end (self-contained loader)
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------
# TRAINING-TIME CUSTOM CLASS
# ---------------------------
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED = [
    "pulse_pressure","map_mean_arterial_pressure","glyco_glu_product",
    "steps_per_wear_day","mvpa_per_wear_day","sedentary_ratio",
    "bmi_age_interaction","steps_per_bmi",
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self): self.feature_names_out_ = None
    def fit(self, X, y=None):
        try: cols = list(X.columns)
        except Exception: cols = []
        self.feature_names_out_ = cols + ENGINEERED
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df_ = X.copy()
        else:
            # fallback if called with numpy
            df_ = pd.DataFrame(X, columns=(self.feature_names_out_ or [])[:X.shape[1]])
        def sdiv(a,b,eps=1e-6): return a/np.maximum(b,eps)
        sbp=pd.to_numeric(df_.get("systolic_bp"),errors="coerce")
        dbp=pd.to_numeric(df_.get("diastolic_bp"),errors="coerce")
        hba=pd.to_numeric(df_.get("hba1c"),errors="coerce")
        glu=pd.to_numeric(df_.get("fasting_glucose"),errors="coerce")
        stp=pd.to_numeric(df_.get("steps_per_week"),errors="coerce")
        mv =pd.to_numeric(df_.get("active_minutes_per_week"),errors="coerce")
        sed=pd.to_numeric(df_.get("sedentary_minutes_per_day"),errors="coerce")
        wrd=pd.to_numeric(df_.get("wear_days"),errors="coerce")
        bmi=pd.to_numeric(df_.get("bmi"),errors="coerce")
        age=pd.to_numeric(df_.get("age"),errors="coerce")
        df_["pulse_pressure"]=sbp-dbp
        df_["map_mean_arterial_pressure"]=dbp+(sbp-dbp)/3.0
        df_["glyco_glu_product"]=hba*glu
        df_["steps_per_wear_day"]=sdiv(stp,wrd)
        df_["mvpa_per_wear_day"]=sdiv(mv,wrd)
        df_["sedentary_ratio"]=sdiv(sed,1440.0)
        df_["bmi_age_interaction"]=bmi*age
        df_["steps_per_bmi"]=sdiv(stp,bmi)
        return df_

# ---------------------------
# PATHS & LOADER (self-contained)
# ---------------------------
def find_base_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here.parent, here, Path.cwd()]
    for base in candidates:
        if (base / "models" / "diabetes_pipeline_sensor.joblib").exists():
            return base
    return Path.cwd()

@st.cache_resource(show_spinner=False)
def load_artifact():
    base = find_base_dir()
    model_path = base / "models" / "diabetes_pipeline_sensor.joblib"
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()
    # Because FeatureEngineer is defined in THIS module (__main__), unpickling works
    art = joblib.load(model_path)
    return art, base

artifact, BASE_DIR = load_artifact()
PIPE = artifact["pipeline"]
DEFAULT_THR = float(artifact.get("threshold", 0.5))
ENG = set(artifact.get("engineered_features", []))
NUM_ALL = list(artifact.get("numeric_features", []))
CAT = list(artifact.get("categorical_features", []))
RAW_NUM = [c for c in NUM_ALL if c not in ENG]
INPUT_COLS = RAW_NUM + CAT

# ---------------------------
# SMALL HELPERS
# ---------------------------
def ensure_columns_for_batch(df_in: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    df = df_in.copy()
    for c in required_cols:
        if c not in df.columns: df[c] = np.nan
    return df[required_cols]

def make_template_csv(required_cols: list[str]) -> str:
    return ",".join(required_cols) + "\n"

def num_input_nan(label: str, help: str = "", value: str = "") -> float | None:
    s = st.text_input(label, value=value, help=help)
    if s.strip() == "": return np.nan
    try: return float(s)
    except: st.warning(f"'{label}' must be numeric (or blank). Using NaN."); return np.nan

def build_df_from_inputs(inputs: dict) -> pd.DataFrame:
    row = {c: np.nan for c in INPUT_COLS}; row.update(inputs)
    return pd.DataFrame([row], columns=INPUT_COLS)

def predict_df(df: pd.DataFrame) -> np.ndarray:
    return PIPE.predict_proba(df)[:, 1]

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Diabetes Detection", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Detection — NHANES + Wearable Sensor")
st.caption("Educational demo. Not medical advice.")

with st.sidebar:
    st.markdown("### Settings")
    thr = st.slider("Decision threshold", 0.05, 0.95, value=float(DEFAULT_THR), step=0.01)
    st.markdown("---")
    st.markdown("**Expected input columns**")
    st.code(", ".join(INPUT_COLS), language="text")

tabs = st.tabs(["🔢 Manual Entry", "📄 Batch CSV", "📊 Model & Plots"])

# Tab 1: Manual
with tabs[0]:
    st.subheader("Manual Entry")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = num_input_nan("Age (years)")
        bmi = num_input_nan("BMI (kg/m²)")
        waist = num_input_nan("Waist circumference (cm)")
        sbp = num_input_nan("Systolic BP (mmHg)")
        dbp = num_input_nan("Diastolic BP (mmHg)")
    with c2:
        hba1c = num_input_nan("HbA1c (%)")
        fpg   = num_input_nan("Fasting glucose (mg/dL)")
        chol  = num_input_nan("Total cholesterol (mg/dL)")
        smoker = st.selectbox("Smoker (0=no, 1=yes)", options=[None, 0, 1], index=0)
    with c3:
        steps_week = num_input_nan("Steps per week")
        mvpa_week  = num_input_nan("Active minutes per week")
        sedentary  = num_input_nan("Sedentary minutes per day")
        wear_days  = num_input_nan("Valid wear days")
        sex = st.selectbox("Sex", options=[None, "male", "female"], index=0)

    X_one = build_df_from_inputs({
        "age": age, "bmi": bmi, "waist_circumference": waist,
        "systolic_bp": sbp, "diastolic_bp": dbp,
        "hba1c": hba1c, "fasting_glucose": fpg, "total_cholesterol": chol,
        "smoker": (None if smoker is None else int(smoker)),
        "steps_per_week": steps_week, "active_minutes_per_week": mvpa_week,
        "sedentary_minutes_per_day": sedentary, "wear_days": wear_days,
        "sex": sex
    })

    if st.button("Predict"):
        try:
            prob = float(predict_df(X_one)[0])
            pred = int(prob >= thr)
            st.success(f"Predicted probability: **{prob:.3f}** → "
                       f"**{'Positive' if pred==1 else 'Negative'}** @ {thr:.2f}")
            with st.expander("Show model input row"):
                st.dataframe(X_one)
        except Exception as e:
            st.error(f"Inference failed: {e}")

# Tab 2: Batch
with tabs[1]:
    st.subheader("Batch Scoring via CSV")
    st.download_button(
        "Download CSV template",
        data=make_template_csv(INPUT_COLS),
        file_name="template_inputs.csv",
        mime="text/csv"
    )
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            st.write("Preview:"); st.dataframe(df_in.head())
            Xb = ensure_columns_for_batch(df_in, INPUT_COLS)
            probs = predict_df(Xb)
            preds = (probs >= thr).astype(int)
            out = df_in.copy()
            out["prob_diabetes"] = probs
            out[f"pred_at_{thr:.2f}"] = preds
            st.write("Results (first 20 rows):")
            st.dataframe(out.head(20))
            st.download_button(
                "Download results CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="diabetes_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Could not score file: {e}")

# Tab 3: Model & Plots
with tabs[2]:
    st.subheader("Model & Evaluation")
    m1, m2, m3 = st.columns(3)
    auc = artifact.get("auc", None); ap = artifact.get("average_precision", None)
    if auc is not None: m1.metric("Validation AUC", f"{auc:.3f}")
    if ap  is not None: m2.metric("Validation AP", f"{ap:.3f}")
    m3.metric("Default threshold", f"{DEFAULT_THR:.2f}")

    st.markdown("#### Feature importance (permutation, validation)")
    fi = artifact.get("feature_importance", {})
    if isinstance(fi, dict) and fi:
        fi_df = pd.DataFrame(sorted(fi.items(), key=lambda kv: kv[1], reverse=True),
                             columns=["feature","importance"]).head(20)
        st.bar_chart(fi_df.set_index("feature"))
        with st.expander("Show top-20 values"):
            st.dataframe(fi_df)
    else:
        st.info("No feature importances found in artifact.")

    st.markdown("#### Evaluation plots (if available)")
    data_dir = BASE_DIR / "data"
    for title, fname in [
        ("ROC Curve", "roc_curve.png"),
        ("Precision–Recall Curve", "pr_curve.png"),
        ("Calibration Curve", "calibration_curve.png"),
        ("Confusion Matrix", "confusion_matrix.png"),
        ("Partial Dependence", "partial_dependence.png"),
        ("SHAP Summary", "shap_summary.png"),
        ("SHAP Dependence", "shap_dependence.png"),
    ]:
        p = data_dir / fname
        if p.exists():
            st.markdown(f"**{title}**")
            st.image(str(p))

st.caption("This app is for educational use only; it does not provide medical advice.")
