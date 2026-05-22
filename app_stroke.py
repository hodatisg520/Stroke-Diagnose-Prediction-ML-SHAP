import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stroke Risk — Clinical ML Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
#  Style: Dashboard — dark sidebar / light content panels
#  Inspired by modern data-viz tools (rtree-viz, Observable, Grafana)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Design tokens ─────────────────────────────────────────────── */
:root {
    /* Surfaces */
    --page-bg:    #F0F2F5;
    --card-bg:    #FFFFFF;
    --card-alt:   #FAFBFC;
    --sidebar-bg: #16181D;
    --sidebar-2:  #1E2028;
    --sidebar-3:  #252830;

    /* Text */
    --t1: #111318;
    --t2: #3D4149;
    --t3: #6B7280;
    --t4: #9CA3AF;
    --t-inv: #F3F4F6;
    --t-inv2: #9BA3B0;

    /* Accent — a teal/cyan that pops without feeling AI-generated */
    --accent:     #0EA5E9;
    --accent-dim: rgba(14,165,233,0.12);
    --accent-glow:rgba(14,165,233,0.25);

    /* Semantic */
    --green:  #10B981;  --green-bg:  rgba(16,185,129,0.10); --green-bd: rgba(16,185,129,0.25);
    --amber:  #F59E0B;  --amber-bg:  rgba(245,158,11,0.10); --amber-bd: rgba(245,158,11,0.25);
    --red:    #EF4444;  --red-bg:    rgba(239,68,68,0.10);   --red-bd:   rgba(239,68,68,0.25);

    /* Borders */
    --bdr:    #E5E7EB;
    --bdr-md: #D1D5DB;

    /* Radius */
    --r-sm: 6px; --r: 10px; --r-lg: 16px; --r-xl: 20px;

    /* Shadows */
    --sh-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --sh:    0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.05);
    --sh-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
}

/* ── Base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── App shell ─────────────────────────────────────────────────── */
.stApp {
    background-color: var(--page-bg) !important;
}
header[data-testid="stHeader"] {
    background-color: var(--sidebar-bg) !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
/* All sidebar text must be light */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    color: var(--t-inv2) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {
    color: var(--t-inv) !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: var(--sidebar-2) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: var(--r-sm) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: var(--t-inv) !important;
}

/* Sidebar slider */
[data-testid="stSidebar"] [data-testid="stSlider"] [class*="stSlider"] div[role="slider"] {
    background-color: var(--accent) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [class*="stSlider"] > div > div {
    background-color: var(--accent) !important;
}

/* Sidebar checkbox / toggle */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: var(--t-inv2) !important;
}

/* Dropdown popover (global) */
[data-baseweb="popover"] > div {
    background-color: #1E2028 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: var(--r) !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4) !important;
}
[data-baseweb="menu"] li {
    color: #D1D5DB !important;
    font-size: 13px !important;
}
[data-baseweb="menu"] li:hover {
    background-color: rgba(255,255,255,0.06) !important;
    color: #F9FAFB !important;
}
[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: var(--accent-dim) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* ── TABS ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--bdr) !important;
    padding: 0 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    margin-bottom: -1px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--t3) !important;
    letter-spacing: 0.01em !important;
    transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--t1) !important; }
.stTabs [aria-selected="true"] {
    color: var(--t1) !important;
    font-weight: 600 !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

/* ── Button ────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 8px rgba(14,165,233,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(14,165,233,0.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Spinner ───────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 99px; }

/* ── Dataframe ─────────────────────────────────────────────────── */
.stDataFrame { border-radius: var(--r) !important; box-shadow: var(--sh-sm) !important; }

/* ════════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ════════════════════════════════════════════════════════════════ */

/* Card */
.cc {
    background: var(--card-bg);
    border: 1px solid var(--bdr);
    border-radius: var(--r-lg);
    padding: 22px 24px;
    box-shadow: var(--sh-sm);
}
.cc-flat {
    background: var(--card-alt);
    border: 1px solid var(--bdr);
    border-radius: var(--r);
    padding: 16px 18px;
}

/* Stat tile */
.stat-tile {
    background: var(--card-bg);
    border: 1px solid var(--bdr);
    border-radius: var(--r-lg);
    padding: 18px 20px;
    box-shadow: var(--sh-sm);
    position: relative;
    overflow: hidden;
}
.stat-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), transparent);
    border-radius: var(--r-lg) var(--r-lg) 0 0;
}
.stat-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--t4);
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-bottom: 8px;
}
.stat-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--t1);
    line-height: 1.1;
    letter-spacing: -0.5px;
    font-variant-numeric: tabular-nums;
}
.stat-sub {
    font-size: 12px;
    color: var(--t4);
    margin-top: 4px;
    font-weight: 400;
}

/* Section title */
.sec-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    font-weight: 600;
    color: var(--t3);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 32px 0 16px;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--bdr);
}

/* Risk badge */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 700;
    border: 1.5px solid;
    letter-spacing: 0.02em;
}

/* Gauge */
.gauge-track {
    width: 100%;
    height: 8px;
    background: #F3F4F6;
    border-radius: 99px;
    overflow: hidden;
    margin: 10px 0 5px;
}
.gauge-fill { height: 100%; border-radius: 99px; transition: width 0.8s ease; }

/* Model inline stats (sidebar) */
.sb-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: rgba(255,255,255,0.06);
    border-radius: var(--r);
    overflow: hidden;
    margin: 10px 0;
}
.sb-cell {
    background: var(--sidebar-2);
    padding: 10px 12px;
}
.sb-cell-label {
    font-size: 9px;
    font-weight: 700;
    color: rgba(156,163,175,0.8);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 3px;
}
.sb-cell-val {
    font-size: 16px;
    font-weight: 700;
    color: #F3F4F6;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.3px;
}
.sb-label {
    font-size: 10px;
    font-weight: 700;
    color: rgba(156,163,175,0.7);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 8px;
    display: block;
}
.sb-divider { border: 0; border-top: 1px solid rgba(255,255,255,0.07); margin: 14px 0; }

/* Finding */
.finding {
    background: var(--card-bg);
    border: 1px solid var(--bdr);
    border-left: 3px solid;
    border-radius: var(--r);
    padding: 13px 16px;
    margin: 8px 0;
    box-shadow: var(--sh-sm);
}
.finding-title { font-size: 13px; font-weight: 600; color: var(--t1); margin: 0 0 4px; }
.finding-body  { font-size: 13px; color: var(--t3); line-height: 1.65; margin: 0; }
.finding-list  { font-size: 13px; color: var(--t3); line-height: 1.7; margin: 6px 0 0; padding-left: 16px; }

/* Methodology row */
.mrow {
    display: flex;
    gap: 16px;
    padding: 9px 0;
    border-bottom: 1px solid var(--bdr);
    align-items: baseline;
    font-size: 13px;
}
.mrow:last-child { border-bottom: none; }
.mrow-key { min-width: 148px; color: var(--t4); font-weight: 500; flex-shrink: 0; }
.mrow-val { color: var(--t2); line-height: 1.5; }

/* Tech chip */
.chip {
    display: inline-flex;
    align-items: center;
    background: var(--page-bg);
    border: 1px solid var(--bdr);
    border-radius: var(--r-sm);
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--t2);
    margin: 3px 3px 3px 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    except FileNotFoundError:
        return None
    df = df.drop(columns=['id'])
    df = df[df['gender'] != 'Other']
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df


@st.cache_data(show_spinner=False)
def preprocess(_df):
    df = _df.copy()
    df['ever_married'] = (df['ever_married'] == 'Yes').astype(int)
    df['gender']       = (df['gender'] == 'Female').astype(int)
    df = pd.get_dummies(df, columns=['work_type', 'Residence_type', 'smoking_status'])
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    return X_res, y_res, X.columns.tolist()


@st.cache_data(show_spinner=False)
def train_all_models(_X, _y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _X, _y, test_size=0.25, random_state=42, stratify=_y)

    defs = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, class_weight='balanced'),
        "Logistic Regression": Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=0.5, random_state=42,
                                       class_weight='balanced'))]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=5,
            random_state=42, class_weight='balanced'),
        "SVM": Pipeline([
            ('sc', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42,
                        class_weight='balanced'))]),
        "KNN": Pipeline([
            ('sc', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=9, weights='distance'))]),
    }

    trained, metrics = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, mdl in defs.items():
        mdl.fit(X_tr, y_tr)
        yp  = mdl.predict(X_te)
        ypr = mdl.predict_proba(X_te)[:, 1]
        cvs = cross_val_score(mdl, _X, _y, cv=cv, scoring='f1', n_jobs=-1)
        fpr, tpr, _ = roc_curve(y_te, ypr)
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_te, yp)  * 100, 1),
            "Precision": round(precision_score(y_te, yp, zero_division=0) * 100, 1),
            "Recall":    round(recall_score(y_te, yp, zero_division=0) * 100, 1),
            "F1":        round(f1_score(y_te, yp, zero_division=0) * 100, 1),
            "AUC-ROC":   round(roc_auc_score(y_te, ypr) * 100, 1),
            "CV F1":     round(cvs.mean() * 100, 1),
            "CV Std":    round(cvs.std()  * 100, 1),
            "roc_fpr":   fpr.tolist(),
            "roc_tpr":   tpr.tolist(),
            "cm":        confusion_matrix(y_te, yp).tolist(),
        }
        trained[name] = mdl

    # Voting Ensemble
    est   = [(n, m) for n, m in trained.items() if n != "SVM"]
    ens   = VotingClassifier(est, voting='soft')
    ens.fit(X_tr, y_tr)
    yp_e  = ens.predict(X_te)
    ypr_e = ens.predict_proba(X_te)[:, 1]
    cvs_e = cross_val_score(ens, _X, _y, cv=cv, scoring='f1', n_jobs=-1)
    fpr_e, tpr_e, _ = roc_curve(y_te, ypr_e)
    metrics["Voting Ensemble"] = {
        "Accuracy":  round(accuracy_score(y_te, yp_e)  * 100, 1),
        "Precision": round(precision_score(y_te, yp_e, zero_division=0) * 100, 1),
        "Recall":    round(recall_score(y_te, yp_e, zero_division=0) * 100, 1),
        "F1":        round(f1_score(y_te, yp_e, zero_division=0) * 100, 1),
        "AUC-ROC":   round(roc_auc_score(y_te, ypr_e) * 100, 1),
        "CV F1":     round(cvs_e.mean() * 100, 1),
        "CV Std":    round(cvs_e.std()  * 100, 1),
        "roc_fpr":   fpr_e.tolist(),
        "roc_tpr":   tpr_e.tolist(),
        "cm":        confusion_matrix(y_te, yp_e).tolist(),
    }
    trained["Voting Ensemble"] = ens
    return trained, metrics, X_tr, X_te, y_tr, y_te


def get_fi(model, feats):
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    if hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf')
        if hasattr(clf, 'coef_'):
            return pd.Series(np.abs(clf.coef_[0]), index=feats).sort_values(ascending=False)
    return None


# ═══════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS — NO margin in base dict (fixes the bug)
# ═══════════════════════════════════════════════════════════════════
def base_layout():
    """Returns base plotly layout kwargs WITHOUT margin (avoid duplicate key bug)."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        font=dict(family='Inter', color='#6B7280', size=12),
        hoverlabel=dict(
            bgcolor='#1E2028',
            bordercolor='rgba(255,255,255,0.12)',
            font=dict(color='#F3F4F6', size=12, family='Inter')),
    )


def ax(title=None, grid=True, **kw):
    """Build axis style dict. titlefont removed — use title=dict(text, font) API (Plotly 5+)."""
    s = dict(
        color='#9CA3AF',
        gridcolor='#F3F4F6',
        linecolor='#E5E7EB',
        tickfont=dict(size=11, color='#9CA3AF', family='Inter'),
        showgrid=grid,
        zeroline=False,
        showline=False,
    )
    if title:
        s['title'] = dict(text=title, font=dict(size=11, color='#9CA3AF', family='Inter'))
    s.update(kw)
    return s


LEG = dict(
    font=dict(color='#374151', size=11, family='Inter'),
    bgcolor='rgba(255,255,255,0.95)',
    bordercolor='#E5E7EB',
    borderwidth=1,
)

# Chart color series — teal/slate palette
C = ['#0EA5E9', '#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']


# ═══════════════════════════════════════════════════════════════════
#  BOOT
# ═══════════════════════════════════════════════════════════════════
df_raw = load_data()
if df_raw is None:
    st.error("Dataset not found — place 'healthcare-dataset-stroke-data.csv' here.")
    st.stop()

with st.spinner("Initialising — training 7 models, approximately 15 s on first run."):
    X_data, y_data, feature_cols = preprocess(df_raw)
    trained_models, all_metrics, X_tr, X_te, y_tr, y_te = train_all_models(X_data, y_data)

MODEL_NAMES = list(trained_models.keys())


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 4px 14px;">
        <div style="font-size:16px;font-weight:700;color:#F9FAFB;letter-spacing:-0.3px;">
            Stroke Risk Assessment
        </div>
        <div style="font-size:11px;color:rgba(156,163,175,0.8);margin-top:3px;
                    font-weight:500;letter-spacing:0.3px;">
            Clinical ML &nbsp;·&nbsp; 7 Classifiers
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)

    st.markdown("<span class='sb-label'>Active Model</span>", unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "model", MODEL_NAMES, label_visibility="collapsed")
    selected_model = trained_models[selected_model_name]

    m = all_metrics[selected_model_name]
    st.markdown(f"""
    <div class="sb-grid">
        <div class="sb-cell">
            <div class="sb-cell-label">Accuracy</div>
            <div class="sb-cell-val">{m['Accuracy']}%</div>
        </div>
        <div class="sb-cell">
            <div class="sb-cell-label">F1 Score</div>
            <div class="sb-cell-val">{m['F1']}%</div>
        </div>
        <div class="sb-cell">
            <div class="sb-cell-label">AUC-ROC</div>
            <div class="sb-cell-val">{m['AUC-ROC']}%</div>
        </div>
        <div class="sb-cell">
            <div class="sb-cell-label">Recall</div>
            <div class="sb-cell-val">{m['Recall']}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)
    st.markdown("<span class='sb-label'>Patient Profile</span>", unsafe_allow_html=True)
    gender    = st.selectbox("Gender", ["Male", "Female"])
    age       = st.slider("Age", 0, 100, 55)
    work_type = st.selectbox("Work Type",
                             ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence = st.selectbox("Residence", ["Urban", "Rural"])

    st.markdown("<span class='sb-label' style='margin-top:14px;'>Clinical Vitals</span>",
                unsafe_allow_html=True)
    avg_glucose = st.slider("Avg. Glucose (mg/dL)", 50.0, 300.0, 105.0, 1.0)
    bmi         = st.slider("BMI (kg/m²)", 10.0, 60.0, 28.5, 0.1)

    st.markdown("<span class='sb-label' style='margin-top:14px;'>Medical History</span>",
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca: hypertension  = st.checkbox("Hypertension")
    with cb: heart_disease = st.checkbox("Heart Disease")
    ever_married = st.toggle("Ever Married", value=True)
    smoking = st.selectbox("Smoking Status",
                           ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    run_btn = st.button("Run Assessment")


# ═══════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════
best_overall = max(all_metrics, key=lambda n: all_metrics[n]['AUC-ROC'])
st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;
            padding:0 0 20px;flex-wrap:wrap;gap:16px;">
    <div>
        <div style="font-size:11px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                    letter-spacing:1.2px;margin-bottom:8px;">
            Medical AI &nbsp;/&nbsp; Stroke Prediction
        </div>
        <h1 style="font-size:1.8rem;font-weight:700;color:#111318;margin:0;
                   letter-spacing:-0.6px;line-height:1.15;">
            Clinical Risk Assessment
        </h1>
        <p style="font-size:14px;color:#6B7280;margin:8px 0 0;max-width:520px;line-height:1.65;">
            Multi-model stroke probability estimation with cross-validation,
            SMOTE class balancing, and SHAP explainability.
        </p>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <div class="cc-flat" style="text-align:center;min-width:80px;">
            <div style="font-size:10px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:0.7px;">Models</div>
            <div style="font-size:22px;font-weight:700;color:#111318;margin-top:2px;">7</div>
        </div>
        <div class="cc-flat" style="text-align:center;min-width:80px;">
            <div style="font-size:10px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:0.7px;">Best AUC</div>
            <div style="font-size:22px;font-weight:700;color:#0EA5E9;margin-top:2px;">
                {all_metrics[best_overall]['AUC-ROC']}%
            </div>
        </div>
        <div class="cc-flat" style="text-align:center;min-width:80px;">
            <div style="font-size:10px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:0.7px;">Records</div>
            <div style="font-size:22px;font-weight:700;color:#111318;margin-top:2px;">5,110</div>
        </div>
    </div>
</div>
<div style="border-top:1px solid #E5E7EB;margin-bottom:24px;"></div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction", "Model Analytics", "Dataset EDA", "Methodology"
])


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — PREDICTION                                             ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab1:
    if not run_btn:
        st.markdown("""
        <div style="padding:56px 0 36px;text-align:center;">
            <div style="width:52px;height:52px;background:linear-gradient(135deg,#EFF6FF,#DBEAFE);
                        border:1px solid #BFDBFE;border-radius:14px;margin:0 auto 20px;
                        display:flex;align-items:center;justify-content:center;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                     stroke="#0EA5E9" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
            </div>
            <h2 style="font-size:1.15rem;font-weight:700;color:#111318;margin:0 0 10px;">
                Ready to Run Assessment
            </h2>
            <p style="font-size:14px;color:#6B7280;max-width:400px;margin:0 auto 40px;line-height:1.65;">
                Complete the patient profile in the sidebar and click
                <strong style="color:#0EA5E9;">Run Assessment</strong> to generate the report.
            </p>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, n, t in [
            (c1, "01", "Enter patient demographics and clinical vitals"),
            (c2, "02", "Select the ML model from the sidebar"),
            (c3, "03", "Click Run Assessment"),
            (c4, "04", "Review score, SHAP values, and clinical findings"),
        ]:
            col.markdown(f"""
            <div class="cc" style="padding:18px;height:100%;">
                <div style="font-size:10px;font-weight:700;color:#0EA5E9;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:8px;">Step {n}</div>
                <div style="font-size:13px;color:#374151;line-height:1.55;">{t}</div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Build prediction ──────────────────────────────────────────
    with st.spinner("Running analysis..."):
        time.sleep(0.3)

    inp_df  = pd.DataFrame({
        'gender': [1 if gender == "Female" else 0], 'age': [age],
        'hypertension': [int(hypertension)], 'heart_disease': [int(heart_disease)],
        'ever_married': [1 if ever_married else 0],
        'avg_glucose_level': [avg_glucose], 'bmi': [bmi],
        'work_type': [work_type], 'Residence_type': [residence], 'smoking_status': [smoking],
    })
    inp_dum = pd.get_dummies(inp_df, columns=['work_type', 'Residence_type', 'smoking_status'])
    inp_fin = inp_dum.reindex(columns=feature_cols, fill_value=0)

    prob     = selected_model.predict_proba(inp_fin)[0][1]
    prob_pct = prob * 100

    if prob < 0.30:
        rl, rc, rbg, rbd = "Low Risk",      "#10B981", "rgba(16,185,129,0.10)",  "rgba(16,185,129,0.3)"
    elif prob < 0.60:
        rl, rc, rbg, rbd = "Moderate Risk", "#F59E0B", "rgba(245,158,11,0.10)", "rgba(245,158,11,0.3)"
    else:
        rl, rc, rbg, rbd = "High Risk",     "#EF4444", "rgba(239,68,68,0.10)",  "rgba(239,68,68,0.3)"

    bmi_cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    glc_cat = "Normal" if avg_glucose < 100 else "Pre-diabetic" if avg_glucose < 126 else "Diabetic range"
    smk_lbl = {"smokes": "Active Smoker", "formerly smoked": "Former Smoker",
                "never smoked": "Non-Smoker", "Unknown": "Unknown"}.get(smoking, "—")
    age_grp = "Young adult" if age < 40 else "Middle-aged" if age < 65 else "Senior"

    # ── Vitals row ────────────────────────────────────────────────
    v1, v2, v3, v4 = st.columns(4)
    for col, lbl, val, sub in [
        (v1, "Age",     str(age),               age_grp),
        (v2, "BMI",     f"{bmi:.1f}",            bmi_cat),
        (v3, "Glucose", f"{avg_glucose:.0f}",   glc_cat),
        (v4, "Smoking", smk_lbl,                 "Status"),
    ]:
        col.markdown(f"""
        <div class="stat-tile">
            <div class="stat-label">{lbl}</div>
            <div class="stat-value">{val}</div>
            <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Risk panel ────────────────────────────────────────────────
    col_risk, col_ring = st.columns([3, 2], gap="medium")

    with col_risk:
        gauge_gradient = (
            f"linear-gradient(90deg, {rc}, {rc}CC)"
            if prob_pct < 60 else
            f"linear-gradient(90deg, #10B981 30%, #F59E0B 60%, {rc})"
        )
        st.markdown(f"""
        <div class="cc">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;
                        flex-wrap:wrap;gap:12px;margin-bottom:20px;">
                <div>
                    <div style="font-size:11px;font-weight:600;color:#9CA3AF;text-transform:uppercase;
                                letter-spacing:0.7px;margin-bottom:8px;">Stroke Probability</div>
                    <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;">
                        <span style="font-size:52px;font-weight:800;color:{rc};letter-spacing:-2px;
                                     line-height:1;font-variant-numeric:tabular-nums;">
                            {prob_pct:.1f}<span style="font-size:28px;font-weight:700;">%</span>
                        </span>
                        <span class="risk-badge" style="color:{rc};background:{rbg};border-color:{rbd};">
                            {rl}
                        </span>
                    </div>
                </div>
            </div>
            <div style="font-size:11px;color:#9CA3AF;font-weight:600;margin-bottom:6px;
                        text-transform:uppercase;letter-spacing:0.5px;">Probability gauge</div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{min(prob_pct,100):.1f}%;background:{gauge_gradient};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;
                        color:#9CA3AF;margin-top:4px;">
                <span>0%</span><span>30%</span><span>60%</span><span>100%</span>
            </div>
            <div style="border-top:1px solid #F3F4F6;margin-top:20px;padding-top:14px;
                        display:flex;gap:20px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:10px;color:#9CA3AF;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:2px;">Model</div>
                    <div style="font-size:13px;font-weight:600;color:#111318;">{selected_model_name}</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#9CA3AF;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:2px;">AUC-ROC</div>
                    <div style="font-size:13px;font-weight:700;color:#0EA5E9;">
                        {all_metrics[selected_model_name]['AUC-ROC']}%
                    </div>
                </div>
                <div>
                    <div style="font-size:10px;color:#9CA3AF;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:2px;">CV F1</div>
                    <div style="font-size:13px;font-weight:600;color:#111318;">
                        {all_metrics[selected_model_name]['CV F1']}% &plusmn; {all_metrics[selected_model_name]['CV Std']}%
                    </div>
                </div>
                <div>
                    <div style="font-size:10px;color:#9CA3AF;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:2px;">Recall</div>
                    <div style="font-size:13px;font-weight:600;color:#111318;">
                        {all_metrics[selected_model_name]['Recall']}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ring:
        fig_ring = go.Figure(go.Pie(
            values=[prob_pct, max(100 - prob_pct, 0)],
            labels=["Risk", "Margin"],
            hole=0.74,
            marker=dict(colors=[rc, "#F0F2F5"], line=dict(color='#FFFFFF', width=4)),
            textinfo='none',
            hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
        ))
        fig_ring.add_annotation(
            text=f"<b>{prob_pct:.1f}%</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=24, color=rc, family="Inter"))
        fig_ring.update_layout(
            showlegend=False,
            margin=dict(l=16, r=16, t=16, b=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=230)
        st.plotly_chart(fig_ring, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Feature Importance & SHAP side by side ────────────────────
    fi = get_fi(selected_model, feature_cols)
    has_fi = fi is not None

    shap_computed = False
    shap_df = None
    if HAS_SHAP and isinstance(selected_model,
                               (RandomForestClassifier, GradientBoostingClassifier,
                                HistGradientBoostingClassifier)):
        try:
            with st.spinner("Computing SHAP values..."):
                exp = shap.TreeExplainer(selected_model)
                svs = exp.shap_values(inp_fin)
                sv  = svs[1][0] if isinstance(svs, list) else svs[0]
            shap_df = (pd.DataFrame({'Feature': feature_cols, 'SHAP': sv})
                       .sort_values('SHAP', key=lambda s: s.abs(), ascending=False)
                       .head(10))
            shap_computed = True
        except Exception:
            pass

    if has_fi or shap_computed:
        col_fi, col_shap = st.columns(2, gap="medium")

        if has_fi:
            with col_fi:
                st.markdown("<div class='sec-title'>Feature Importance</div>", unsafe_allow_html=True)
                top10 = fi.head(10)
                fig_fi = go.Figure(go.Bar(
                    x=top10.values, y=top10.index,
                    orientation='h',
                    marker=dict(
                        color=top10.values,
                        colorscale=[[0, '#DBEAFE'], [1, '#0EA5E9']],
                        showscale=False,
                        line=dict(width=0)),
                    text=[f"{v:.3f}" for v in top10.values],
                    textposition='outside',
                    textfont=dict(size=10, color='#9CA3AF', family='JetBrains Mono'),
                    hovertemplate='%{y}: %{x:.4f}<extra></extra>',
                ))
                fig_fi.update_layout(
                    **base_layout(),
                    title=None,
                    xaxis=ax(grid=True),
                    yaxis=ax(autorange='reversed', grid=False),
                    margin=dict(l=8, r=64, t=8, b=8),
                    height=320)
                st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})

        if shap_computed and shap_df is not None:
            with col_shap:
                st.markdown("<div class='sec-title'>SHAP — Per-Prediction Attribution</div>",
                            unsafe_allow_html=True)
                s_colors = ['#EF4444' if v > 0 else '#10B981' for v in shap_df['SHAP']]
                fig_shap = go.Figure(go.Bar(
                    x=shap_df['SHAP'], y=shap_df['Feature'],
                    orientation='h',
                    marker=dict(color=s_colors, line=dict(width=0)),
                    text=[f"{v:+.3f}" for v in shap_df['SHAP']],
                    textposition='outside',
                    textfont=dict(size=10, color='#9CA3AF', family='JetBrains Mono'),
                    hovertemplate='%{y}: %{x:+.4f}<extra></extra>',
                ))
                fig_shap.add_vline(x=0, line_color='#E5E7EB', line_width=1.5)
                fig_shap.update_layout(
                    **base_layout(),
                    title=None,
                    xaxis=ax(title="SHAP value", grid=True),
                    yaxis=ax(autorange='reversed', grid=False),
                    margin=dict(l=8, r=64, t=8, b=8),
                    height=320)
                st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
                st.markdown("""
                <p style="font-size:11px;color:#9CA3AF;margin:-4px 0 0 4px;">
                    Red = increases stroke risk &nbsp;|&nbsp; Green = reduces stroke risk
                </p>""", unsafe_allow_html=True)

    # ── Clinical Findings ─────────────────────────────────────────
    st.markdown("<div class='sec-title'>Clinical Findings</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cc" style="margin-bottom:14px;">
        <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
            <div style="background:{rbg};border:1.5px solid {rbd};border-radius:10px;
                        padding:12px 18px;flex-shrink:0;">
                <div style="font-size:10px;font-weight:700;color:{rc};text-transform:uppercase;
                            letter-spacing:0.7px;">Overall Assessment</div>
                <div style="font-size:17px;font-weight:700;color:{rc};margin-top:3px;">{rl}</div>
            </div>
            <div style="font-size:13px;color:#6B7280;line-height:1.7;padding-top:4px;">
                {selected_model_name} estimates a stroke probability of
                <strong style="color:#111318;">{prob_pct:.1f}%</strong> for this patient profile.
                This result is for research and educational purposes only.
                Consult a qualified clinician before any clinical decision.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def mk_finding(title, body, bullets=None, color="#EF4444"):
        li = ""
        if bullets:
            li = "<ul class='finding-list'>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
        return f"""
        <div class="finding" style="border-left-color:{color};">
            <div class="finding-title" style="color:{color};">{title}</div>
            <p class="finding-body">{body}</p>
            {li}
        </div>"""

    fd = []
    if avg_glucose > 200:
        fd.append(mk_finding(f"Critically Elevated Glucose — {avg_glucose:.0f} mg/dL",
            "Hyperglycaemic range substantially amplifies cerebrovascular risk.",
            ["Endocrinology referral within 24 hours.", "Eliminate refined carbohydrates.",
             "Blood glucose monitoring 3–4× daily."], "#EF4444"))
    elif avg_glucose > 140:
        fd.append(mk_finding(f"Elevated Glucose — {avg_glucose:.0f} mg/dL",
            "Pre-diabetic range. Sustained elevation damages vascular endothelium.",
            ["Reduce refined carbohydrate intake.", "30 min moderate exercise daily.",
             "Schedule HbA1c with GP."], "#F59E0B"))
    else:
        fd.append(mk_finding("Glucose Within Normal Range",
            "Blood glucose is well-controlled. Maintain current dietary habits.", color="#10B981"))

    if bmi >= 30:
        fd.append(mk_finding(f"Obesity — BMI {bmi:.1f}",
            "Obesity raises hypertension, insulin resistance, and stroke risk significantly.",
            ["Target BMI <25 through gradual reduction.", "Caloric deficit with adequate protein.",
             "Aerobic exercise 5× per week, 30–45 min."], "#EF4444"))
    elif bmi > 25:
        fd.append(mk_finding(f"Overweight — BMI {bmi:.1f}",
            "Modest weight reduction substantially lowers cardiovascular event risk.",
            ["Target 8,000–10,000 steps daily.", "Increase protein; reduce ultra-processed foods."],
            "#F59E0B"))
    else:
        fd.append(mk_finding(f"Healthy Weight — BMI {bmi:.1f}",
            "Weight is within the healthy range.", color="#10B981"))

    if hypertension:
        fd.append(mk_finding("Hypertension Present",
            "The leading modifiable stroke risk factor.",
            ["Verify medication adherence.", "Sodium <1,500 mg/day.",
             "Reduce alcohol; stress-management strategies."], "#EF4444"))

    if heart_disease:
        fd.append(mk_finding("Cardiac Condition Noted",
            "Cardiac pathology markedly increases thromboembolic stroke risk.",
            ["Discuss anticoagulation with cardiologist.", "Regular ECG monitoring recommended."],
            "#EF4444"))

    if smoking == 'smokes':
        fd.append(mk_finding("Active Smoker",
            "Smoking approximately doubles stroke risk.",
            ["Cessation is the highest-impact modifiable intervention.",
             "Nicotine replacement therapy or varenicline are first-line options.",
             "Risk normalises within 2–5 years post-cessation."], "#EF4444"))
    elif smoking == 'formerly smoked':
        fd.append(mk_finding("Former Smoker",
            "Residual risk decreases each year of maintained cessation.", color="#F59E0B"))

    if age >= 65:
        fd.append(mk_finding(f"Advanced Age — {age} years",
            "Stroke risk doubles per decade after age 55.",
            ["Annual cardiovascular review recommended.", "Maintain physical activity."],
            "#F59E0B"))

    if not fd:
        fd.append(mk_finding("No Major Risk Factors Identified",
            "Vitals are within healthy ranges. Continue routine check-ups.", color="#10B981"))

    for b in fd:
        st.markdown(b, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:18px;padding:12px 16px;background:#F9FAFB;border:1px solid #E5E7EB;
                border-radius:8px;">
        <span style="font-size:11px;color:#9CA3AF;line-height:1.6;">
            <strong style="color:#6B7280;">Disclaimer.</strong>
            For educational and research use only. Generated by {selected_model_name}
            (AUC-ROC {all_metrics[selected_model_name]['AUC-ROC']}%, SMOTE-balanced training set,
            5-fold CV). Not a substitute for clinical evaluation by a licensed healthcare professional.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — MODEL ANALYTICS                                        ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab2:
    MK  = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    df_m = pd.DataFrame({
        n: {k: v[k] for k in MK + ["CV F1", "CV Std"]}
        for n, v in all_metrics.items()
    }).T.reset_index().rename(columns={'index': 'Model'})
    best = df_m.loc[df_m['AUC-ROC'].idxmax(), 'Model']

    # ── Top-line summary ──────────────────────────────────────────
    st.markdown("<div class='sec-title' style='margin-top:0;'>Performance Summary</div>",
                unsafe_allow_html=True)
    scols = st.columns(len(MODEL_NAMES))
    for ci, (col, name) in enumerate(zip(scols, MODEL_NAMES)):
        mv  = all_metrics[name]
        hi  = name == best
        col.markdown(f"""
        <div class="cc" style="text-align:center;padding:14px 10px;
                               {'border:1.5px solid #0EA5E9;' if hi else ''}">
            <div style="font-size:10px;font-weight:700;color:{'#0EA5E9' if hi else '#9CA3AF'};
                        text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
            <div style="font-size:20px;font-weight:800;color:{'#0EA5E9' if hi else '#111318'};
                        font-variant-numeric:tabular-nums;letter-spacing:-0.5px;">{mv['AUC-ROC']}%</div>
            <div style="font-size:11px;color:#9CA3AF;margin-top:2px;">AUC-ROC</div>
            <div style="font-size:12px;color:#374151;margin-top:8px;">F1 {mv['F1']}%</div>
            {'<div style="font-size:10px;color:#0EA5E9;font-weight:700;margin-top:4px;">BEST</div>' if hi else ''}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Radar + leaderboard ───────────────────────────────────────
    st.markdown("<div class='sec-title'>Detailed Comparison</div>", unsafe_allow_html=True)
    col_r, col_t = st.columns([4, 6], gap="medium")

    with col_r:
        radar_traces = []
        for i, row in df_m.iterrows():
            is_b = row['Model'] == best
            rv = [row[k] for k in MK] + [row[MK[0]]]
            radar_traces.append(go.Scatterpolar(
                r=rv, theta=MK + [MK[0]], fill='toself',
                name=row['Model'],
                line=dict(color=C[i % len(C)], width=2.5 if is_b else 1.5),
                fillcolor=C[i % len(C)],
                opacity=0.12 if not is_b else 0.20,
            ))
        fig_radar = go.Figure(radar_traces)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor='#F0F2F5', tickfont=dict(size=9, color='#9CA3AF'),
                                linecolor='#E5E7EB'),
                angularaxis=dict(gridcolor='#F0F2F5',
                                 tickfont=dict(size=11, color='#374151', family='Inter')),
                bgcolor='#FFFFFF'),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(**LEG, orientation='v', x=1.05),
            margin=dict(l=20, r=100, t=16, b=16),
            height=380,
            font=dict(family='Inter', color='#374151'))
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    with col_t:
        df_disp = df_m.set_index('Model')[MK + ["CV F1", "CV Std"]]
        st.dataframe(
            df_disp.style
            .background_gradient(cmap='Blues', vmin=50, vmax=100, subset=MK)
            .format("{:.1f}%")
            .set_properties(**{
                'font-family': '"JetBrains Mono", monospace',
                'font-size': '12px',
                'color': '#111318',
            })
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#F9FAFB'),
                          ('color', '#6B7280'),
                          ('font-size', '11px'),
                          ('font-weight', '600'),
                          ('text-transform', 'uppercase'),
                          ('letter-spacing', '0.5px')],
            }]),
            use_container_width=True, height=380)

    # ── ROC Curves ────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>ROC Curves</div>", unsafe_allow_html=True)
    fig_roc = go.Figure()
    fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
                      line=dict(color='#E5E7EB', dash='dot', width=1.5))
    for i, (name, mv) in enumerate(all_metrics.items()):
        is_b = name == best
        fig_roc.add_trace(go.Scatter(
            x=mv['roc_fpr'], y=mv['roc_tpr'],
            mode='lines', name=f"{name}  ({mv['AUC-ROC']}%)",
            line=dict(color=C[i % len(C)], width=2.5 if is_b else 1.5),
            opacity=1 if is_b else 0.7,
        ))
    fig_roc.update_layout(
        **base_layout(),
        xaxis=ax("False Positive Rate", range=[0, 1]),
        yaxis=ax("True Positive Rate", range=[0, 1]),
        legend=dict(**LEG, x=0.52, y=0.06),
        margin=dict(l=44, r=16, t=16, b=44),
        height=420)
    st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': False})

    # ── Confusion Matrices ────────────────────────────────────────
    st.markdown("<div class='sec-title'>Confusion Matrices</div>", unsafe_allow_html=True)
    rows_cm = [MODEL_NAMES[i:i+3] for i in range(0, len(MODEL_NAMES), 3)]
    for row_g in rows_cm:
        cols_cm = st.columns(len(row_g))
        for col_cm, mn in zip(cols_cm, row_g):
            cm = np.array(all_metrics[mn]['cm'])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=['Pred: No Stroke', 'Pred: Stroke'],
                y=['True: No Stroke', 'True: Stroke'],
                colorscale=[[0, '#EFF6FF'], [0.5, '#38BDF8'], [1, '#0369A1']],
                showscale=False,
                text=cm, texttemplate='<b>%{text}</b>',
                textfont=dict(size=18, color='#1E293B'),
                hovertemplate='%{y} / %{x}: %{z}<extra></extra>',
            ))
            fig_cm.update_layout(
                **base_layout(),
                title=dict(text=mn, font=dict(size=12, color='#111318', weight=600),
                           x=0, xanchor='left'),
                xaxis=dict(tickfont=dict(size=9, color='#6B7280')),
                yaxis=dict(tickfont=dict(size=9, color='#6B7280')),
                margin=dict(l=8, r=8, t=36, b=8),
                height=220)
            col_cm.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

    # ── CV F1 Bar ─────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>5-Fold Cross-Validation F1</div>", unsafe_allow_html=True)
    cv_m = [all_metrics[n]['CV F1']  for n in MODEL_NAMES]
    cv_s = [all_metrics[n]['CV Std'] for n in MODEL_NAMES]
    fig_cv = go.Figure(go.Bar(
        x=MODEL_NAMES, y=cv_m,
        error_y=dict(type='data', array=cv_s, visible=True,
                     color='#9CA3AF', thickness=1.5, width=6),
        marker=dict(
            color=[C[0] if n == best else '#CBD5E1' for n in MODEL_NAMES],
            line=dict(width=0)),
        text=[f"{v:.1f}%" for v in cv_m],
        textposition='outside',
        textfont=dict(size=11, color='#374151', family='JetBrains Mono'),
        hovertemplate='%{x}: %{y:.1f}% ± %{error_y.array:.1f}%<extra></extra>',
    ))
    fig_cv.update_layout(
        **base_layout(),
        yaxis=ax("CV F1 Score (%)", range=[0, 115]),
        xaxis=ax(grid=False),
        margin=dict(l=44, r=16, t=12, b=12),
        height=300)
    st.plotly_chart(fig_cv, use_container_width=True, config={'displayModeBar': False})


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — DATASET EDA                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab3:
    total    = len(df_raw)
    n_stroke = int(df_raw['stroke'].sum())
    n_hlt    = total - n_stroke

    st.markdown("<div class='sec-title' style='margin-top:0;'>Dataset Overview</div>",
                unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.columns(5)
    for col, lbl, val, sub in [
        (d1, "Total Records",  f"{total:,}",         "patients"),
        (d2, "Stroke Cases",   f"{n_stroke:,}",      f"{n_stroke/total*100:.1f}% prevalence"),
        (d3, "Healthy",        f"{n_hlt:,}",         f"{n_hlt/total*100:.1f}%"),
        (d4, "Mean Age",       f"{df_raw['age'].mean():.1f}","years"),
        (d5, "Mean BMI",       f"{df_raw['bmi'].mean():.1f}","kg/m²"),
    ]:
        col.markdown(f"""
        <div class="stat-tile">
            <div class="stat-label">{lbl}</div>
            <div class="stat-value">{val}</div>
            <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        fig_age = go.Figure()
        for sv, cl, lb in [(0, '#CBD5E1', 'No Stroke'), (1, '#0EA5E9', 'Stroke')]:
            fig_age.add_trace(go.Histogram(
                x=df_raw[df_raw['stroke'] == sv]['age'],
                name=lb, marker_color=cl, opacity=0.8,
                nbinsx=30, histnorm='probability density'))
        fig_age.update_layout(
            **base_layout(),
            title=dict(text='Age Distribution by Outcome',
                       font=dict(size=13, color='#111318'), x=0),
            barmode='overlay',
            xaxis=ax('Age'), yaxis=ax('Density'),
            legend=dict(**LEG),
            margin=dict(l=40, r=16, t=40, b=36), height=300)
        st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})

    with e2:
        fig_glc = go.Figure()
        for sv, cl, lb in [(0, '#CBD5E1', 'No Stroke'), (1, '#6366F1', 'Stroke')]:
            fig_glc.add_trace(go.Histogram(
                x=df_raw[df_raw['stroke'] == sv]['avg_glucose_level'],
                name=lb, marker_color=cl, opacity=0.8,
                nbinsx=35, histnorm='probability density'))
        fig_glc.update_layout(
            **base_layout(),
            title=dict(text='Glucose Distribution by Outcome',
                       font=dict(size=13, color='#111318'), x=0),
            barmode='overlay',
            xaxis=ax('Avg Glucose (mg/dL)'), yaxis=ax('Density'),
            legend=dict(**LEG),
            margin=dict(l=40, r=16, t=40, b=36), height=300)
        st.plotly_chart(fig_glc, use_container_width=True, config={'displayModeBar': False})

    e3, e4 = st.columns(2)
    with e3:
        fig_pie = go.Figure(go.Pie(
            values=[n_hlt, n_stroke], labels=['No Stroke', 'Stroke'], hole=0.68,
            marker=dict(colors=['#CBD5E1', '#0EA5E9'], line=dict(color='#FFFFFF', width=3)),
            textinfo='label+percent', textfont=dict(size=12, color='#374151'),
            hovertemplate='%{label}: %{value:,} (%{percent})<extra></extra>',
        ))
        fig_pie.add_annotation(text=f"<b>{total:,}</b>", x=0.5, y=0.5, showarrow=False,
                               font=dict(size=20, color='#111318', family='Inter'))
        fig_pie.update_layout(
            **base_layout(),
            title=dict(text='Class Distribution (Before SMOTE)',
                       font=dict(size=13, color='#111318'), x=0),
            legend=dict(**LEG),
            margin=dict(l=12, r=12, t=40, b=12), height=300)
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with e4:
        wt = (df_raw.groupby('work_type')['stroke'].mean() * 100).reset_index()
        wt.columns = ['Type', 'Rate']
        wt = wt.sort_values('Rate')
        fig_wt = go.Figure(go.Bar(
            x=wt['Rate'], y=wt['Type'], orientation='h',
            marker=dict(
                color=wt['Rate'],
                colorscale=[[0, '#DBEAFE'], [1, '#0EA5E9']],
                showscale=False, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in wt['Rate']],
            textposition='outside',
            textfont=dict(size=11, color='#6B7280', family='JetBrains Mono'),
            hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
        ))
        fig_wt.update_layout(
            **base_layout(),
            title=dict(text='Stroke Rate by Occupation',
                       font=dict(size=13, color='#111318'), x=0),
            xaxis=ax('Stroke Rate (%)'),
            yaxis=ax(grid=False),
            margin=dict(l=12, r=56, t=40, b=36), height=300)
        st.plotly_chart(fig_wt, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='sec-title'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
    corr = df_raw[['age', 'bmi', 'avg_glucose_level',
                   'hypertension', 'heart_disease', 'stroke']].corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, '#EFF6FF'], [0.5, '#38BDF8'], [1, '#0369A1']],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate='<b>%{text}</b>', textfont=dict(size=12, color='#1E293B'),
        hovertemplate='%{y} / %{x}: %{z:.3f}<extra></extra>',
    ))
    fig_corr.update_layout(
        **base_layout(),
        xaxis=dict(tickfont=dict(size=11, color='#6B7280')),
        yaxis=dict(tickfont=dict(size=11, color='#6B7280')),
        margin=dict(l=8, r=8, t=8, b=8), height=380)
    st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='sec-title'>BMI vs. Glucose — Risk Scatter</div>", unsafe_allow_html=True)
    fig_sc = go.Figure()
    for sv, cl, sym, lb in [(0, '#CBD5E1', 'circle', 'No Stroke'),
                             (1, '#EF4444', 'x-thin', 'Stroke')]:
        n = min(500, int((df_raw['stroke'] == sv).sum()))
        sub = df_raw[df_raw['stroke'] == sv].sample(n, random_state=1)
        fig_sc.add_trace(go.Scatter(
            x=sub['bmi'], y=sub['avg_glucose_level'],
            mode='markers', name=lb, opacity=0.65,
            marker=dict(color=cl, size=5, symbol=sym,
                        line=dict(color='rgba(0,0,0,0.1)', width=0.5))))
    fig_sc.update_layout(
        **base_layout(),
        xaxis=ax('BMI'), yaxis=ax('Avg Glucose (mg/dL)'),
        legend=dict(**LEG),
        margin=dict(l=44, r=16, t=16, b=44), height=380)
    st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — METHODOLOGY                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown("""
    <div class="cc" style="margin-bottom:20px;">
        <h2 style="font-size:1.1rem;font-weight:700;color:#111318;margin:0 0 10px;
                   letter-spacing:-0.2px;">Project Methodology</h2>
        <p style="font-size:14px;color:#6B7280;line-height:1.7;margin:0;">
            A production-quality binary classification pipeline demonstrating data science competency
            across preprocessing, multi-model benchmarking, rigorous evaluation, and
            prediction-level explainability — designed as a portfolio piece.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2, gap="medium")
    with m1:
        st.markdown("""
        <div class="cc">
            <div style="font-size:10px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:0.8px;margin-bottom:16px;">Dataset</div>
        """, unsafe_allow_html=True)
        for k, v in [("Source", "Kaggle Stroke Prediction Dataset"),
                     ("Records", "5,110 patients"),
                     ("Features", "10 clinical input variables"),
                     ("Target", "Binary — stroke / no stroke"),
                     ("Class imbalance", "~4.87% positive class")]:
            st.markdown(f"""
            <div class="mrow"><span class="mrow-key">{k}</span>
            <span class="mrow-val">{v}</span></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="cc" style="margin-top:14px;">
            <div style="font-size:10px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:0.8px;margin-bottom:16px;">Preprocessing Pipeline</div>
        """, unsafe_allow_html=True)
        for i, (k, v) in enumerate([
            ("Missing values",   "BMI median imputation"),
            ("Encoding",         "Binary + one-hot for categoricals"),
            ("SMOTE",            "Synthetic minority oversampling"),
            ("StandardScaler",   "Pipeline-scoped for SVM, LR, KNN"),
            ("Train / Test",     "75% / 25%, stratified split"),
        ], 1):
            st.markdown(f"""
            <div class="mrow">
                <span class="mrow-key" style="display:flex;align-items:center;gap:8px;">
                    <span style="display:inline-flex;align-items:center;justify-content:center;
                                 width:20px;height:20px;background:#EFF6FF;border:1px solid #BFDBFE;
                                 border-radius:99px;font-size:11px;font-weight:700;color:#0EA5E9;
                                 flex-shrink:0;">{i}</span>{k}</span>
                <span class="mrow-val">{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="cc">
            <div style="font-size:10px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:0.8px;margin-bottom:16px;">Models Implemented</div>
        """, unsafe_allow_html=True)
        for i, (name, desc) in enumerate([
            ("Random Forest",          "200-tree ensemble. Non-linear, native feature importances."),
            ("Logistic Regression",    "L2-regularised linear baseline (C=0.5). Interpretable."),
            ("Gradient Boosting",      "Sequential residual correction. Strong on tabular data."),
            ("Hist Gradient Boosting", "Histogram-based boosting. Handles missing values natively."),
            ("SVM",                    "RBF-kernel max-margin classifier. Calibrated probabilities."),
            ("KNN",                    "Distance-weighted voting (k=9). Non-parametric baseline."),
            ("Voting Ensemble",        "Soft-vote meta-learner over 5 base classifiers."),
        ]):
            dot_color = C[i % len(C)]
            st.markdown(f"""
            <div class="mrow" style="align-items:flex-start;">
                <span class="mrow-key" style="display:flex;align-items:center;gap:8px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:{dot_color};
                                 flex-shrink:0;margin-top:3px;display:inline-block;"></span>
                    <strong style="color:#374151;">{name}</strong>
                </span>
                <span class="mrow-val">{desc}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cc">
        <div style="font-size:10px;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                    letter-spacing:0.8px;margin-bottom:18px;">Evaluation Strategy</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:28px;">
            <div>
                <div style="font-size:13px;font-weight:600;color:#111318;margin-bottom:6px;">
                    Why not Accuracy alone?
                </div>
                <div style="font-size:13px;color:#6B7280;line-height:1.7;">
                    With ~5% positive rate, always predicting "No Stroke" yields 95% accuracy
                    but zero clinical value. Primary metrics:
                    <strong style="color:#374151;">Recall</strong> and
                    <strong style="color:#374151;">AUC-ROC</strong>.
                </div>
            </div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#111318;margin-bottom:6px;">
                    5-Fold Stratified CV
                </div>
                <div style="font-size:13px;color:#6B7280;line-height:1.7;">
                    Each fold preserves class ratio. Mean ± std F1 measures robustness,
                    not single-split optimism. Reported alongside test-set metrics.
                </div>
            </div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#111318;margin-bottom:6px;">
                    SHAP Explainability
                </div>
                <div style="font-size:13px;color:#6B7280;line-height:1.7;">
                    SHapley Additive exPlanations attribute each feature's marginal contribution
                    to individual predictions — making black-box models clinically interpretable.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>Technology Stack</div>", unsafe_allow_html=True)
    tech = [("Python 3.10+", "Runtime"), ("Streamlit", "Web framework"),
            ("scikit-learn", "ML & pipelines"), ("imbalanced-learn", "SMOTE"),
            ("SHAP", "Explainability"), ("Plotly", "Charts"), ("Pandas", "Data"),
            ("NumPy", "Numerics")]
    chips = "".join(
        f"<span class='chip'><strong style='color:#111318;margin-right:6px;'>{n}</strong>"
        f"<span style='color:#9CA3AF;font-size:11px;'>{r}</span></span>"
        for n, r in tech)
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;'>{chips}</div>",
                unsafe_allow_html=True)