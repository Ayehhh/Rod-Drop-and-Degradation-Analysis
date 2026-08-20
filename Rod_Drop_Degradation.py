import os
import io
import shutil
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import brentq, curve_fit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Set Streamlit Page Config
st.set_page_config(
    page_title="Rod Drop Prognostic Analysis",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Reciprocating Compressor Rod Drop Degradation and Prognostic Analysis")
st.markdown("This tool models the physical wear rate of compressor rider rings (Degradation Analysis) and uses predictive models to project the exact date when rod drop will exceed safety thresholds (Prognostic Analysis).")

# ==========================================
# 1. DATA SOURCE & PARAMETERS CONFIGURATION
# ==========================================
st.sidebar.header("1. Data Source")
data_source = st.sidebar.selectbox(
    "Choose Analysis Mode:",
    ["User Specified", "Sample Data"]
)

df = None

if data_source == "User Specified":
    st.sidebar.subheader("2. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset (.xlsx / .xls)", type=["xlsx", "xls"])

    st.sidebar.subheader("3. Engineering Parameters")
    NEW_CLEARANCE = st.sidebar.number_input("As-New Clearance [mm]", value=2.000, step=0.001, format="%.3f")
    BN_L_THRESHOLD = st.sidebar.number_input("Bently Nevada L Alarm Threshold [µm]", value=-1500.0, step=10.0, format="%.1f")
    BN_LL_THRESHOLD = st.sidebar.number_input("Bently Nevada LL Alarm Threshold [µm]", value=-2000.0, step=10.0, format="%.1f")
    MIN_CLEARANCE = st.sidebar.number_input("Minimum Allowable Clearance / OEM Limit [mm]", value=0.500, step=0.001, format="%.3f")
    CONFIDENCE_PCT = st.sidebar.number_input("Confidence Level Analysis [%]", value=95.0, min_value=50.0, max_value=99.9, step=1.0)

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        st.info("👈 Please upload an Excel dataset in the sidebar to begin analysis.")
        st.stop()

else:  # Sample Data Mode
    st.sidebar.success("✅ Running with Synthetic Historical Data")

    st.sidebar.subheader("Adjust Thresholds (Editable)")
    NEW_CLEARANCE = st.sidebar.number_input("As-New Clearance [mm]", value=2.000, step=0.001, format="%.3f")
    BN_L_THRESHOLD = st.sidebar.number_input("Bently Nevada L Alarm Threshold [µm]", value=-1500.0, step=10.0, format="%.1f")
    BN_LL_THRESHOLD = st.sidebar.number_input("Bently Nevada LL Alarm Threshold [µm]", value=-2000.0, step=10.0, format="%.1f")
    MIN_CLEARANCE = st.sidebar.number_input("Minimum Allowable Clearance / OEM Limit [mm]", value=0.500, step=0.001, format="%.3f")
    CONFIDENCE_PCT = st.sidebar.number_input("Confidence Level Analysis [%]", value=95.0, min_value=50.0, max_value=99.9, step=1.0)

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=18, freq='30D')
    days_passed = np.arange(18) * 30
    synthetic_wear = 0.05 * (days_passed ** 1.3) + np.random.normal(0, 15, 18)
    raw_readings = -np.abs(synthetic_wear)

    df = pd.DataFrame({
        "timestamp": dates,
        "raw_um": raw_readings
    })

# Setup Output Directory (Clean directory to keep ZIP lightweight)
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

latest_wear_um = df["wear_um"].iloc[-1]
latest_clearance_mm = df["clearance_mm"].iloc[-1]

# ==========================================
# 3. REGRESSION MODELING (INCL. LOG-NORMAL)
# ==========================================
def _lin(x, a, b): return a * x + b
def _quad(x, a, b, c): return a * x**2 + b * x + c
def _power(x, a, b): return a * np.power(np.maximum(x, 1e-6), b)
def _expo(x, a, b): return a * np.exp(np.clip(b * x, -100, 100))
def _logf(x, a, b): return a * np.log(x + 1.0) + b
def _lognorm(x, a, shape, scale): return a * stats.lognorm.cdf(np.maximum(x, 1e-6), s=shape, scale=scale)

MODELS = {
    "Linear": (_lin, [1.0, 0.0]),
    "Quadratic": (_quad, [0.01, 1.0, 0.0]),
    "Power": (_power, [1.0, 1.0]),
    "Exponential": (_expo, [1.0, 0.01]),
    "Logarithmic": (_logf, [1.0, 0.0]),
    "Log-Normal": (_lognorm, [np.max(wear_um) * 1.5, 0.8, np.mean(days) + 1.0])
}

model_results = {}
for name, (func, p0) in MODELS.items():
    try:
        popt, _ = curve_fit(func, days, wear_um, p0=p0, maxfev=10000)
        y_pred = func(days, *popt)
        ss_res = np.sum((wear_um - y_pred)**2)
        ss_tot = np.sum((wear_um - np.mean(wear_um))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        dof = max(len(days) - len(popt), 1)
        resid_std = np.sqrt(ss_res / dof)
        model_results[name] = {
            "func": func, "popt": popt, "r2": r2,
            "resid_std": resid_std, "dof": dof, "ss_res": ss_res
        }
    except Exception:
        pass

best_name = max(model_results, key=lambda k: model_results[k]["r2"])
best = model_results[best_name]

# ==========================================
# 4. PROGNOSTIC BREACH CALCULATION
# ==========================================
st.subheader("📊 Model Comparison & Fit Metrics")
model_comparison_data = []
for name, res in model_results.items():
    model_comparison_data.append({
        "Model Name": name,
        "R² Score": f"{res['r2']:.4f}",
        "Residual Std (µm)": f"{res['resid_std']:.2f}",
        "Status": "✅ Selected (Best Fit)" if name == best_name else "Candidate"
    })
st.dataframe(pd.DataFrame(model_comparison_data), use_container_width=True)

def solve_crossing(model, target_wear, conf_pct, max_days=3650):
    func, popt, dof, std = model["func"], model["popt"], model["dof"], model["resid_std"]
    t_val = stats.t.ppf((1 + conf_pct / 100.0) / 2, dof) if dof >= 1 else 0.0
    band = t_val * std

    def get_date(f):
        xs = np.linspace(0, max_days, 15000)
        ys = f(xs)
        idx = np.where(np.diff(np.sign(ys)) != 0)[0]
        if len(idx) == 0: return None
        try: return brentq(f, xs[idx[0]], xs[idx[0]+1])
        except: return None

    early = get_date(lambda x: func(x, *popt) + band - target_wear)
    central = get_date(lambda x: func(x, *popt) - target_wear)
    late = get_date(lambda x: func(x, *popt) - band - target_wear)
    return early, central, late

max_horizon = max(days[-1] * 4, days[-1] + 3650)
f_L = solve_crossing(best, WEAR_TARGET_L, CONFIDENCE_PCT, max_horizon)
f_LL = solve_crossing(best, WEAR_TARGET_LL, CONFIDENCE_PCT, max_horizon)
f_Min = solve_crossing(best, WEAR_TARGET_MIN, CONFIDENCE_PCT, max_horizon)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Model", f"{best_name}", f"R² = {best['r2']:.4f}")
m2.metric("Latest Wear", f"{latest_wear_um:.1f} µm")
m3.metric("Current Clearance", f"{latest_clearance_mm:.3f} mm")
m4.metric("Confidence Level", f"{CONFIDENCE_PCT:.1f}%")

st.subheader("📋 Prognostic Breach & RUL Projection Summary")
prognosis_data = []
latest_day = days[-1]

targets_info = [
    ("L Alarm", BN_L_THRESHOLD, f_L, CLEARANCE_AT_L),
    ("LL Alarm", BN_LL_THRESHOLD, f_LL, CLEARANCE_AT_LL),
    ("OEM Min Limit", -WEAR_TARGET_MIN, f_Min, MIN_CLEARANCE)
]

for label, raw_alarm, (e, c, l), target_c in targets_info:
    c_date = (t0 + timedelta(days=c)).strftime('%Y-%m-%d') if c else "N/A"
    e_date = (t0 + timedelta(days=e)).strftime('%Y-%m-%d') if e else "N/A"
    l_date = (t0 + timedelta(days=l)).strftime('%Y-%m-%d') if l else "N/A"
    rul_days = f"{int(c - latest_day)} Days" if c and c >= latest_day else "Exceeded / N/A"

    prognosis_data.append({
        "Threshold Level": label,
        "Alarm Threshold (um)": f"{raw_alarm:.1f}",
        "Target Clearance (mm)": f"{target_c:.3f}",
        "Earliest Date": e_date,
        "Expected Date": c_date,
        "Latest Date": l_date,
        "RUL (Days)": rul_days
    })

st.table(pd.DataFrame(prognosis_data))

# ==========================================
# 5. GENERATE VISUALIZATION (TREND ONLY)
# ==========================================
st.subheader("📈 Prognostic Trend Plot")

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.scatter(df["timestamp"], df["clearance_mm"], color="#1f77b4", s=25, alpha=0.8, label="Measured Observations")

candidate_days = [d for d in [f_L[1], f_LL[1], f_Min[1]] if d is not None]
x_max_plot = max(candidate_days) * 1.15 if candidate_days else days[-1] * 2
x_plot = np.linspace(0, x_max_plot, 500)
y_wear_plot = best["func"](x_plot, *best["popt"])
y_clear_plot = NEW_CLEARANCE - (y_wear_plot / 1000.0)
dates_plot = [t0 + timedelta(days=d) for d in x_plot]

t_val = stats.t.ppf((1 + CONFIDENCE_PCT / 100.0) / 2, best["dof"]) if best["dof"] >= 1 else 0.0
band_mm = (t_val * best["resid_std"]) / 1000.0

ax.plot(dates_plot, y_clear_plot, color="#d62728", linewidth=2, label=f"Best Fit ({best_name})")
ax.fill_between(dates_plot, y_clear_plot - band_mm, y_clear_plot + band_mm, color="#d62728", alpha=0.15, label=f"{CONFIDENCE_PCT:.0f}% CI")

ax.axhline(CLEARANCE_AT_L, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"L Alarm ({CLEARANCE_AT_L:.3f} mm)")
ax.axhline(CLEARANCE_AT_LL, color="#d62728", linestyle="--", linewidth=1.5, label=f"LL Alarm ({CLEARANCE_AT_LL:.3f} mm)")
ax.axhline(MIN_CLEARANCE, color="black", linestyle=":", linewidth=1.5, label=f"Min Clearance ({MIN_CLEARANCE:.3f} mm)")

ax.set_ylabel("Clearance (mm)")
ax.set_xlabel("Date")
ax.set_title("Piston Rod Clearance Prognostic Trend", fontweight="bold")
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.6)
fig.autofmt_xdate()
fig.tight_layout()

plot_img_path = os.path.join(OUTPUT_DIR, "clearance_trend_plot_en.png")
fig.savefig(plot_img_path, dpi=150, bbox_inches="tight")
st.pyplot(fig)

# ==========================================
# 6. PDF REPORT GENERATION & ZIP PACKAGING
# ==========================================
pdf_file_path = os.path.join(OUTPUT_DIR, f"Piston_Rod_Clearance_Prognostic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

def generate_pdf_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor('#1B365D'), alignment=1, spaceAfter=12)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#FFFFFF'), backColor=colors.HexColor('#4A777A'), spaceBefore=10, spaceAfter=6, leftIndent=4)

    story.append(Paragraph("DEGRADATION AND PROGNOSTIC ANALYSIS REPORT FOR ROD DROP AND ESTIMATED CLEARANCE", title_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("TECHNICAL SPECIFICATIONS & THRESHOLDS", section_style))
    spec_data = [
        ["Parameter", "Value", "Unit"],
        ["As-Left Bottom Piston-to-Liner Clearance", f"{NEW_CLEARANCE:.3f}", "mm"],
        ["Bently Nevada L Alarm Threshold", f"{BN_L_THRESHOLD:.1f}", "um"],
        ["Bently Nevada LL Alarm Threshold", f"{BN_LL_THRESHOLD:.1f}", "um"],
        ["Minimum Allowable Clearance (OEM)", f"{MIN_CLEARANCE:.3f}", "mm"],
        ["Calculated L Clearance Limit", f"{CLEARANCE_AT_L:.3f}", "mm"],
        ["Calculated LL Clearance Limit", f"{CLEARANCE_AT_LL:.3f}", "mm"],
        ["Statistical Confidence Level", f"{CONFIDENCE_PCT:.1f}", "%"]
    ]
    t_spec = Table(spec_data, colWidths=[270, 130, 100])
    t_spec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t_spec)
    story.append(Spacer(1, 10))

    story.append(Paragraph("PROGNOSTIC BREACH PROJECTION SUMMARY", section_style))
    prog_headers = ["Threshold Level", "Alarm Threshold (um)", "Target Clearance (mm)", "Earliest Date", "Expected Date", "Latest Date"]
    prog_table_data = [prog_headers]
    
    for row in prognosis_data:
        prog_table_data.append([
            row["Threshold Level"], 
            row["Alarm Threshold (um)"], 
            row["Target Clearance (mm)"], 
            row["Earliest Date"], 
            row["Expected Date"], 
            row["Latest Date"]
        ])

    t_prog = Table(prog_table_data, colWidths=[100, 85, 85, 75, 75, 80])
    t_prog.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t_prog)
    story.append(Spacer(1, 10))

    story.append(Paragraph("PROGNOSTIC TREND VISUALISATION", section_style))
    story.append(Spacer(1, 4))
    story.append(RLImage(plot_img_path, width=500, height=230))

    doc.build(story)

generate_pdf_report(pdf_file_path)

# Archive PDF and Image into ZIP
zip_base_name = os.path.join(os.getcwd(), "Prognosis_Output_Files_Package")
zip_archive_path = shutil.make_archive(zip_base_name, 'zip', OUTPUT_DIR)

with open(pdf_file_path, "rb") as f:
    pdf_bytes = f.read()

with open(zip_archive_path, "rb") as f:
    zip_bytes = f.read()

# Download Buttons Section
st.subheader("📥 Download Prognosis Reports")
dcol1, dcol2 = st.columns(2)
dcol1.download_button(
    label="📄 Download Comprehensive PDF Report",
    data=pdf_bytes,
    file_name=os.path.basename(pdf_file_path),
    mime="application/pdf"
)
dcol2.download_button(
    label="📦 Download Complete ZIP Package (PDF + Image)",
    data=zip_bytes,
    file_name=f"Prognosis_Output_Files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
    mime="application/zip"
)
