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

# ==========================================
# 1. DATA SOURCE & PARAMETERS CONFIGURATION
# ==========================================
st.sidebar.header("1. Data Source")
data_source = st.sidebar.selectbox(
    "Choose Analysis Mode:",
    ["User Specified (File Upload)", "Copy & Paste Bulk Data", "Sample Data"]
)

df = None

if data_source == "User Specified (File Upload)":
    st.sidebar.subheader("2. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset (.xlsx / .xls)", type=["xlsx", "xls"])

    st.sidebar.subheader("3. Engineering Parameters")
    NEW_CLEARANCE = st.sidebar.number_input("As-New Clearance [mm]", value=2.000, step=0.001, format="%.3f")
    BN_L_THRESHOLD = st.sidebar.number_input("Bently Nevada L Alarm Threshold [µm]", value=-1500.0, step=10.0, format="%.1f")
    BN_LL_THRESHOLD = st.sidebar.number_input("Bently Nevada LL Alarm Threshold [µm]", value=-2000.0, step=10.0, format="%.1f")
    MIN_CLEARANCE = st.sidebar.number_input("Minimum Allowable Clearance / OEM Limit [mm]", value=0.300, step=0.001, format="%.3f")
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
    NEW_CLEARANCE = st.sidebar.number_input("As-New Clearance [mm]", value=2.500, step=0.001, format="%.3f")
    BN_L_THRESHOLD = st.sidebar.number_input("Bently Nevada L Alarm Threshold [µm]", value=-1500.0, step=10.0, format="%.1f")
    BN_LL_THRESHOLD = st.sidebar.number_input("Bently Nevada LL Alarm Threshold [µm]", value=-2000.0, step=10.0, format="%.1f")
    MIN_CLEARANCE = st.sidebar.number_input("Minimum Allowable Clearance / OEM Limit [mm]", value=0.300, step=0.001, format="%.3f")
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
    
    # Accelerated wear progression reaching ~1100 um near the end
    target_total_wear = 1100.0  # Brings clearance down to 2.5 - 1.1 = 1.4 mm
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

latest_raw_um = df["raw_um"].iloc[-1]
latest_wear_um = df["wear_um"].iloc[-1]
latest_clearance_mm = df["clearance_mm"].iloc[-1]
min_hist_clearance_mm = df["clearance_mm"].min()

# ==========================================
# 3. EXPANDED REGRESSION MODELING SUITE
# ==========================================
def _lin(x, a, b): 
    return a * x + b

def _quad(x, a, b, c): 
    return a * x**2 + b * x + c

def _power(x, a, b): 
    return a * np.power(np.maximum(x, 1e-6), b)

def _expo(x, a, b): 
    return a * np.exp(np.clip(b * x, -100, 100))

def _logf(x, a, b): 
    return a * np.log(x + 1.0) + b

def _lognorm(x, a, shape, scale): 
    return a * stats.lognorm.cdf(np.maximum(x, 1e-6), s=shape, scale=scale)

def _weibull(x, a, beta, eta): 
    return a * (1.0 - np.exp(-1.0 * (np.maximum(x, 1e-6) / eta)**beta))

def _loglogis(x, a, alpha, beta):
    xs = np.maximum(x, 1e-6)
    return a * (1.0 / (1.0 + (xs / alpha)**(-beta)))

max_w = max(np.max(wear_um) * 2.5, 3000.0)
mean_d = max(np.mean(days), 1.0)

MODELS = {
    "Linear": (_lin, [1.0, 0.0], (-np.inf, np.inf)),
    "Quadratic": (_quad, [0.0001, 0.1, 0.0], (-np.inf, np.inf)),
    "Power Law": (_power, [0.1, 1.2], (0, np.inf)),
    "Exponential": (_expo, [10.0, 0.001], (-np.inf, np.inf)),
    "Logarithmic": (_logf, [100.0, 0.0], (-np.inf, np.inf)),
    "Log-Normal CDF": (_lognorm, [max_w, 1.0, mean_d], ([0, 0.01, 0.1], [max_w * 5, 10.0, 50000])),
    "Weibull CDF": (_weibull, [max_w, 1.5, mean_d], ([0, 0.1, 0.1], [max_w * 5, 10.0, 50000])),
    "Log-Logistic CDF": (_loglogis, [max_w, mean_d, 1.5], ([0, 0.1, 0.1], [max_w * 5, 50000, 10.0]))
}

model_results = {}
for name, (func, p0, bnds) in MODELS.items():
    try:
        popt, _ = curve_fit(func, days, wear_um, p0=p0, bounds=bnds, maxfev=30000)
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

# --- MODEL SELECTION BY USER ---
st.sidebar.header("4. Model Selection")
auto_best = max(model_results, key=lambda k: model_results[k]["r2"])
model_options = ["Auto (Select Best R²)"] + list(model_results.keys())
selected_model_option = st.sidebar.selectbox("Regression Model Choice:", model_options)

if selected_model_option == "Auto (Select Best R²)":
    best_name = auto_best
else:
    best_name = selected_model_option

best = model_results[best_name]

# ==========================================
# 4. EXPLANATION & FIT METRICS DISPLAY
# ==========================================
st.subheader("📊 Model Comparison & Fit Metrics")

with st.expander("💡 Technical Guidance: Metrics, RUL & Model Selection Guide"):
    st.markdown("""
    * **Remaining Useful Life (RUL):** The estimated operational time remaining before rider ring wear breaches an alarm threshold or minimum clearance limit.
      $$\\text{RUL (Days)} = \\text{Expected Breach Date} - \\text{Last Observed Data Date}$$
    * **Residual Standard Deviation (Residual Std in µm):** Measures the physical noise or distance between actual rod drop readings and fitted model predictions. Lower values signify higher accuracy.
    * **$R^2$ Score (Coefficient of Determination):** Quantifies how well the variance in degradation is captured by the time variable ($1.0$ is perfect fit).
    * **Model Selection Recommendation:** 
        * **Power Law / Linear / Quadratic:** Recommended for steady mechanical wear scenarios where friction steadily increases over operational hours.
        * **Exponential:** Best suited for severe wear acceleration prior to total material fatigue.
        * **CDF Models (Log-Normal, Weibull, Log-Logistic):** Useful when rider ring wear decelerates after initial bedding-in.
    """)

model_comparison_data = []
for name, res in model_results.items():
    model_comparison_data.append({
        "Model Name": name,
        "R² Score": f"{res['r2']:.4f}",
        "Residual Std (µm)": f"{res['resid_std']:.2f}",
        "Status": "✅ Selected" if name == best_name else ("Auto Best" if name == auto_best else "Candidate")
    })
st.dataframe(pd.DataFrame(model_comparison_data), use_container_width=True)

# ==========================================
# 5. PROGNOSTIC BREACH CALCULATION & METRICS
# ==========================================
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

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Selected Model", f"{best_name}", f"R² = {best['r2']:.4f}")
m2.metric("Current Average Probe Position", f"{latest_raw_um:.1f} µm")
m3.metric("Current Clearance", f"{latest_clearance_mm:.3f} mm")
m4.metric("Min Historical Clearance", f"{min_hist_clearance_mm:.3f} mm")
m5.metric("Confidence Level", f"{CONFIDENCE_PCT:.1f}%")

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
# 6. VISUALIZATION WITH BREACH MARKERS
# ==========================================
candidate_days = [d for d in [f_L[1], f_LL[1], f_Min[1]] if d is not None]
x_max_plot = max(candidate_days) * 1.15 if candidate_days else days[-1] * 2
x_plot = np.linspace(0, x_max_plot, 500)
y_wear_plot = best["func"](x_plot, *best["popt"])
y_clear_plot = NEW_CLEARANCE - (y_wear_plot / 1000.0)
dates_plot = [t0 + timedelta(days=d) for d in x_plot]

t_val = stats.t.ppf((1 + CONFIDENCE_PCT / 100.0) / 2, best["dof"]) if best["dof"] >= 1 else 0.0
band_mm = (t_val * best["resid_std"]) / 1000.0

# Breach Points Pre-processing for Annotations
breach_events = [
    ("L Alarm", f_L[1], CLEARANCE_AT_L, "#ff7f0e"),
    ("LL Alarm", f_LL[1], CLEARANCE_AT_LL, "#d62728"),
    ("Min Limit", f_Min[1], MIN_CLEARANCE, "black")
]

# --- 6A. STATIC GRAPH FOR PDF REPORT & ZIP ARCHIVE (MATPLOTLIB) ---
fig_static, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.scatter(df["timestamp"], df["clearance_mm"], color="#1f77b4", s=25, alpha=0.8, label="Measured Observations")
ax.plot(dates_plot, y_clear_plot, color="#d62728", linewidth=2, label=f"Fit ({best_name})")
ax.fill_between(dates_plot, y_clear_plot - band_mm, y_clear_plot + band_mm, color="#d62728", alpha=0.15, label=f"{CONFIDENCE_PCT:.0f}% CI")

ax.axhline(CLEARANCE_AT_L, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"L Alarm ({CLEARANCE_AT_L:.3f} mm)")
ax.axhline(CLEARANCE_AT_LL, color="#d62728", linestyle="--", linewidth=1.5, label=f"LL Alarm ({CLEARANCE_AT_LL:.3f} mm)")
ax.axhline(MIN_CLEARANCE, color="black", linestyle=":", linewidth=1.5, label=f"Min Clearance ({MIN_CLEARANCE:.3f} mm)")

# Add Callout Markers for Matplotlib Plot
for name, breach_d, clear_val, color in breach_events:
    if breach_d is not None:
        b_date = t0 + timedelta(days=breach_d)
        ax.plot(b_date, clear_val, marker='o', markersize=7, color=color, markeredgecolor='black')
        ax.annotate(
            f"{name}\n{b_date.strftime('%Y-%m-%d')}",
            xy=(b_date, clear_val),
            xytext=(15, 15),
            textcoords='offset points',
            fontsize=7,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color, lw=1)
        )

ax.set_ylabel("Clearance (mm)")
ax.set_xlabel("Date")
ax.set_title(f"Piston Rod Clearance Prognostic Trend ({best_name})", fontweight="bold")
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.6)
fig_static.autofmt_xdate()
fig_static.tight_layout()

# Save image file for ReportLab PDF generation
plot_img_path = os.path.join(OUTPUT_DIR, "clearance_trend_plot_en.png")
fig_static.savefig(plot_img_path, dpi=150, bbox_inches="tight")
plt.close(fig_static)

# --- 6B. INTERACTIVE GRAPH FOR STREAMLIT UI (PLOTLY) ---
fig_interactive = go.Figure()

# 1. Measured Observations
fig_interactive.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["clearance_mm"],
    mode='markers',
    name='Measured Observations',
    marker=dict(color='#1f77b4', size=8)
))

# 2. Lower Confidence Band
fig_interactive.add_trace(go.Scatter(
    x=dates_plot,
    y=y_clear_plot - band_mm,
    mode='lines',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False,
    hoverinfo="skip"
))

# 3. Upper Confidence Band
fig_interactive.add_trace(go.Scatter(
    x=dates_plot,
    y=y_clear_plot + band_mm,
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(214, 39, 40, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name=f"{CONFIDENCE_PCT:.0f}% Confidence Interval",
    hoverinfo="skip"
))

# 4. Fitted Line
fig_interactive.add_trace(go.Scatter(
    x=dates_plot,
    y=y_clear_plot,
    mode='lines',
    name=f'Model ({best_name})',
    line=dict(color='#d62728', width=2)
))

# 5. Threshold Lines
fig_interactive.add_hline(y=CLEARANCE_AT_L, line_dash="dash", line_color="#ff7f0e", annotation_text=f"L Alarm ({CLEARANCE_AT_L:.3f} mm)", annotation_position="bottom right")
fig_interactive.add_hline(y=CLEARANCE_AT_LL, line_dash="dash", line_color="#d62728", annotation_text=f"LL Alarm ({CLEARANCE_AT_LL:.3f} mm)", annotation_position="bottom right")
fig_interactive.add_hline(y=MIN_CLEARANCE, line_dash="dot", line_color="black", annotation_text=f"Min Clearance ({MIN_CLEARANCE:.3f} mm)", annotation_position="bottom right")

# 6. Add Callout Markers & Annotations on Interactive Graph
for name, breach_d, clear_val, color in breach_events:
    if breach_d is not None:
        b_date = t0 + timedelta(days=breach_d)
        fig_interactive.add_trace(go.Scatter(
            x=[b_date],
            y=[clear_val],
            mode='markers+text',
            name=f'Breach: {name}',
            marker=dict(color=color, size=11, symbol='diamond', line=dict(color='black', width=1)),
            text=[f"<b>{name} Breach</b><br>{b_date.strftime('%Y-%m-%d')}"],
            textposition="top right",
            hoverinfo="text"
        ))

fig_interactive.update_layout(
    title=dict(text=f"<b>Piston Rod Clearance Prognostic Trend ({best_name})</b>", x=0.5),
    xaxis_title="Date",
    yaxis_title="Clearance (mm)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    margin=dict(l=40, r=40, t=50, b=50),
    template="plotly_white"
)

st.plotly_chart(fig_interactive, use_container_width=True)

# ==========================================
# 7. REPORT GENERATION
# ==========================================
pdf_file_path = os.path.join(OUTPUT_DIR, f"Piston_Rod_Clearance_Prognostic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

def generate_pdf_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor('#1B365D'), alignment=1, spaceAfter=12)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#FFFFFF'), backColor=colors.HexColor('#4A777A'), spaceBefore=10, spaceAfter=6, leftIndent=4)

    hdr_style = ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.whitesmoke, alignment=1)
    hdr_style_l = ParagraphStyle('THL', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.whitesmoke, alignment=0)
    body_style = ParagraphStyle('TD', fontName='Helvetica', fontSize=8, leading=10, alignment=1)
    body_style_l = ParagraphStyle('TDL', fontName='Helvetica', fontSize=8, leading=10, alignment=0)

    story.append(Paragraph("DEGRADATION AND PROGNOSTIC ANALYSIS REPORT FOR ROD DROP AND ESTIMATED CLEARANCE", title_style))
    story.append(Spacer(1, 6))

    # SECTION 1: TECHNICAL SPECIFICATIONS
    story.append(Paragraph("TECHNICAL SPECIFICATIONS & THRESHOLDS", section_style))
    spec_data = [
        [Paragraph("Parameter", hdr_style_l), Paragraph("Value", hdr_style), Paragraph("Unit", hdr_style)],
        [Paragraph("As-Left Bottom Piston-to-Liner Clearance", body_style_l), Paragraph(f"{NEW_CLEARANCE:.3f}", body_style), Paragraph("mm", body_style)],
        [Paragraph("Bently Nevada L Alarm Threshold", body_style_l), Paragraph(f"{BN_L_THRESHOLD:.1f}", body_style), Paragraph("um", body_style)],
        [Paragraph("Bently Nevada LL Alarm Threshold", body_style_l), Paragraph(f"{BN_LL_THRESHOLD:.1f}", body_style), Paragraph("um", body_style)],
        [Paragraph("Minimum Allowable Clearance (OEM)", body_style_l), Paragraph(f"{MIN_CLEARANCE:.3f}", body_style), Paragraph("mm", body_style)],
        [Paragraph("Calculated L Clearance Limit", body_style_l), Paragraph(f"{CLEARANCE_AT_L:.3f}", body_style), Paragraph("mm", body_style)],
        [Paragraph("Calculated LL Clearance Limit", body_style_l), Paragraph(f"{CLEARANCE_AT_LL:.3f}", body_style), Paragraph("mm", body_style)],
        [Paragraph("Statistical Confidence Level", body_style_l), Paragraph(f"{CONFIDENCE_PCT:.1f}", body_style), Paragraph("%", body_style)]
    ]
    t_spec = Table(spec_data, colWidths=[270, 130, 100])
    t_spec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_spec)
    story.append(Spacer(1, 8))

    # SECTION 2: MODEL COMPARISON
    story.append(Paragraph("MODEL COMPARISON & FIT METRICS", section_style))
    comp_headers = [
        Paragraph("Model Name", hdr_style_l), 
        Paragraph("R² Score", hdr_style), 
        Paragraph("Residual Std (um)", hdr_style), 
        Paragraph("Status", hdr_style)
    ]
    comp_table_data = [comp_headers]
    for row in model_comparison_data:
        comp_table_data.append([
            Paragraph(row["Model Name"], body_style_l),
            Paragraph(row["R² Score"], body_style),
            Paragraph(row["Residual Std (µm)"], body_style),
            Paragraph(row["Status"], body_style)
        ])

    t_comp = Table(comp_table_data, colWidths=[130, 100, 120, 150])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))

    # SECTION 3: PROGNOSTIC BREACH
    story.append(Paragraph("PROGNOSTIC BREACH PROJECTION SUMMARY", section_style))
    prog_headers = [
        Paragraph("Threshold Level", hdr_style_l), 
        Paragraph("Alarm Threshold (um)", hdr_style), 
        Paragraph("Target Clearance (mm)", hdr_style), 
        Paragraph("Earliest Date", hdr_style), 
        Paragraph("Expected Date", hdr_style), 
        Paragraph("Latest Date", hdr_style)
    ]
    prog_table_data = [prog_headers]
    
    for row in prognosis_data:
        prog_table_data.append([
            Paragraph(row["Threshold Level"], body_style_l), 
            Paragraph(row["Alarm Threshold (um)"], body_style), 
            Paragraph(row["Target Clearance (mm)"], body_style), 
            Paragraph(row["Earliest Date"], body_style), 
            Paragraph(row["Expected Date"], body_style), 
            Paragraph(row["Latest Date"], body_style)
        ])

    t_prog = Table(prog_table_data, colWidths=[90, 85, 85, 80, 80, 80])
    t_prog.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_prog)

    story.append(PageBreak())

    # SECTION 4 (PAGE 2): VISUALIZATION
    story.append(Paragraph(f"PROGNOSTIC TREND VISUALISATION ({best_name})", section_style))
    story.append(Spacer(1, 10))
    story.append(RLImage(plot_img_path, width=500, height=250))

    doc.build(story)

generate_pdf_report(pdf_file_path)

# Archive ZIP Package
zip_base_name = os.path.join(os.getcwd(), "Prognosis_Output_Files_Package")
zip_archive_path = shutil.make_archive(zip_base_name, 'zip', OUTPUT_DIR)

with open(pdf_file_path, "rb") as f:
    pdf_bytes = f.read()

with open(zip_archive_path, "rb") as f:
    zip_bytes = f.read()

# Download Section
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
