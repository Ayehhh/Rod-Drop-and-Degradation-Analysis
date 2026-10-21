import os
import io
import shutil
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import brentq, curve_fit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Set Streamlit Page Config
st.set_page_config(
    page_title="Rod Drop Prognostic Analysis",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Reciprocating Compressor Rod Drop Degradation and Prognostic Analysis")
st.markdown("This tool models the physical wear rate of compressor rider rings (Degradation Analysis) and uses predictive mathematical models to project the exact date when rod drop will exceed safety thresholds (Prognostic Analysis).")
st.caption(
    "Developed by Harith Irfan. Contact me for further information"
    " @ mharithirfan14@icloud.com")

# ==========================================
# 1. DATA SOURCE & PARAMETERS CONFIGURATION
# ==========================================
st.sidebar.header("1. Asset Information & Data Source")

# Added Complex & Asset Tag Inputs
COMPLEX_NAME = st.sidebar.text_input("Complex / Plant Name", value="Complex A")
ASSET_TAG = st.sidebar.text_input("Asset Tag / Name", value="C1234")

asset_title_suffix = f" - {ASSET_TAG}" if ASSET_TAG.strip() else ""

data_source = st.sidebar.selectbox(
    "Choose Analysis Mode:",
    ["User Specified (File Upload)", "Copy & Paste Bulk Data", "Sample Data"]
)

df = None

if data_source == "User Specified (File Upload)":
    st.sidebar.subheader("2. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset (.xlsx / .xls)", type=["xlsx", "xls"])

    st.sidebar.subheader("3. Engineering Parameters")
    NEW_CLEARANCE = st.sidebar.number_input("As-New Clearance [mm]", value=2.000, step=0.001, format="%.3g")
    BN_L_THRESHOLD = st.sidebar.number_input("Bently Nevada L Alarm Threshold [µm]", value=-1500.0, step=10.0, format="%.1f")
    BN_LL_THRESHOLD = st.sidebar.number_input("Bently Nevada LL Alarm Threshold [µm]", value=-2000.0, step=10.0, format="%.1f")
    MIN_CLEARANCE = st.sidebar.number_input("Minimum Allowable Clearance / OEM Limit [mm]", value=0.300, step=0.001, format="%.3g")
    CONFIDENCE_PCT = st.sidebar.number_input("Confidence Level Analysis [%]", value=95.0, min_value=50.0, max_value=99.9, step=1.0)

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        st.info("👈 Please upload an Excel dataset in the sidebar to begin analysis.")
        st.stop()

elif data_source == "Copy & Paste Bulk Data":
    st.sidebar.subheader("2. Paste Data")
    st.sidebar.caption("Paste two columns from Excel (Timestamp / Date and Raw Probe Reading in µm).")
    pasted_data = st.sidebar.text_area(
        "Paste Raw Data Here (Tab or Comma separated):",
        height=180,
        placeholder="2025-01-01\t0.0\n2025-02-01\t-120.5\n2025-03-01\t-250.0..."
    )

    st.sidebar.subheader("3. Engineering Parameters")
    NEW_CLEARANCE = st.sidebar.number_input("As-New Clearance [mm]", value=2.500, step=0.001, format="%.2f")
    BN_L_THRESHOLD = st.sidebar.number_input("Bently Nevada L Alarm Threshold [µm]", value=-1500.0, step=10.0, format="%.1f")
    BN_LL_THRESHOLD = st.sidebar.number_input("Bently Nevada LL Alarm Threshold [µm]", value=-2000.0, step=10.0, format="%.1f")
    MIN_CLEARANCE = st.sidebar.number_input("Minimum Allowable Clearance / OEM Limit [mm]", value=0.300, step=0.001, format="%.2f")
    CONFIDENCE_PCT = st.sidebar.number_input("Confidence Level Analysis [%]", value=95.0, min_value=50.0, max_value=99.9, step=1.0)

    if pasted_data.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted_data), sep=r'\s+|,', engine='python', header=None)
        except Exception as e:
            st.error(f"Error parsing pasted data: {e}")
            st.stop()
    else:
        st.info("👈 Please paste tabular data in the sidebar text box to proceed.")
        st.stop()

else:  # Sample Data Mode
    st.sidebar.success("✅ Running with Synthetic Historical Data (Worst Case Progression)")

    st.sidebar.subheader("Adjust Thresholds (Editable)")
    NEW_CLEARANCE = st.sidebar.number_input("As-New Clearance [mm]", value=2.500, step=0.001, format="%.3f")
    BN_L_THRESHOLD = st.sidebar.number_input("Bently Nevada L Alarm Threshold [µm]", value=-1500.0, step=10.0, format="%.1f")
    BN_LL_THRESHOLD = st.sidebar.number_input("Bently Nevada LL Alarm Threshold [µm]", value=-2000.0, step=10.0, format="%.1f")
    MIN_CLEARANCE = st.sidebar.number_input("Minimum Allowable Clearance / OEM Limit [mm]", value=0.300, step=0.001, format="%.3g")
    CONFIDENCE_PCT = st.sidebar.number_input("Confidence Level Analysis [%]", value=95.0, min_value=50.0, max_value=99.9, step=1.0)

    # Generate Synthetic Data degrading from ~2.5 mm down to ~1.4 mm clearance (~1100 um total wear)
    np.random.seed(42)
    n_points = 20
    dates = pd.date_range(end=datetime.now(), periods=n_points, freq='30D')
    days_passed = np.arange(n_points) * 30
    
    target_total_wear = 1100.0
    base_wear = target_total_wear * (days_passed / days_passed[-1]) ** 1.35
    noise = np.random.normal(0, 12, n_points)
    synthetic_wear = np.maximum(0, base_wear + noise)
    raw_readings = -synthetic_wear

    df = pd.DataFrame({
        "timestamp": dates,
        "raw_um": raw_readings
    })

# Setup Output Directory
OUTPUT_DIR = "Prognosis_Output_Files"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLEARANCE_AT_L = NEW_CLEARANCE - (abs(BN_L_THRESHOLD) / 1000.0)
CLEARANCE_AT_LL = NEW_CLEARANCE - (abs(BN_LL_THRESHOLD) / 1000.0)
WEAR_TARGET_L = abs(BN_L_THRESHOLD)
WEAR_TARGET_LL = abs(BN_LL_THRESHOLD)
WEAR_TARGET_MIN = (NEW_CLEARANCE - MIN_CLEARANCE) * 1000.0

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
if len(df.columns) >= 2:
    df = df.iloc[:, :2]
    df.columns = ["timestamp", "raw_um"]

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["raw_um"] = pd.to_numeric(df["raw_um"], errors="coerce")
df = df.dropna().sort_values("timestamp").reset_index(drop=True)

df["wear_um"] = -df["raw_um"]
df["clearance_mm"] = NEW_CLEARANCE - (df["wear_um"] / 1000.0)

t0 = df["timestamp"].iloc[0]
days = (df["timestamp"] - t0).dt.total_seconds().values / 86400.0
wear_um = df["wear_um"].values

latest_timestamp = df["timestamp"].iloc[-1].strftime('%Y-%m-%d')
latest_raw_um = df["raw_um"].iloc[-1]
latest_wear_um = df["wear_um"].iloc[-1]
latest_clearance_mm = df["clearance_mm"].iloc[-1]
min_hist_clearance_mm = df["clearance_mm"].min()

# ==========================================
# 3. EXPANDED REGRESSION MODELING SUITE
# ==========================================
def _lin(x, a, b): return a * x + b
def _quad(x, a, b, c): return a * x**2 + b * x + c
def _power(x, a, b): return a * np.power(np.maximum(x, 1e-6), b)
def _expo(x, a, b): return a * np.exp(np.clip(b * x, -100, 100))
def _logf(x, a, b): return a * np.log(x + 1.0) + b
def _lognorm(x, a, shape, scale): return a * stats.lognorm.cdf(np.maximum(x, 1e-6), s=shape, scale=scale)
def _weibull(x, a, beta, eta): return a * (1.0 - np.exp(-1.0 * (np.maximum(x, 1e-6) / eta)**beta))
def _loglogis(x, a, alpha, beta): return a * (1.0 / (1.0 + (np.maximum(x, 1e-6) / alpha)**(-beta)))

max_w = max(np.max(wear_um) * 2.5, 3000.0)
mean_d = max(np.mean(days), 1.0)

MODELS = {
    "Linear": (_lin, [1.0, 0.0], (-np.inf, np.inf)),
    "Quadratic": (_quad, [0.0001, 0.1, 0.0], (-np.inf, np.inf)),
    "Power Law": (_power, [0.1, 1.2], (0, np.inf)),
    "Exponential": (_expo, [10.0, 0.001], (-np.inf, np.inf)),
    "Logarithmic": (_logf, [100.0, 0.0], (-np.inf, np.inf)),
    "Log-Normal CDF": (_lognorm, [max_w, 1.0, mean_d], ([0, 0.01, 0.1], [max_w * 5, 10.0, 50000])),
    "Weibull CDF":
