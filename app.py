"""
Alzheimer's Risk Screening: interview demo app.

Four panels, following how the question actually gets worked:
  Score a patient     live risk score for an entered record, across four models
  Set the cut-off     where the screening threshold belongs, and what it costs
  What drives it      which features carry the signal, by two methods
  Where it tops out   whether more modelling is worth it, tested rather than assumed

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

# Executive palette: one navy accent, muted slate for everything secondary,
# and a muted red / teal pair reserved for risk meaning only.
BG = "#F6F7F9"
CARD = "#FFFFFF"
INK = "#0F1B2D"
INK_2 = "#5B6675"
INK_3 = "#8A94A3"
GRID = "#E9ECF1"
LINE = "#E3E7ED"
ACCENT = "#1F4E8C"        # navy: the recommended model, primary series
ACCENT_SOFT = "#DCE6F2"
SLATE = "#AEB7C4"         # every non-highlighted series
RISK = "#B42318"          # at risk / missed case
SAFE = "#0E7C66"          # not at risk / correctly cleared
SURFACE = CARD
POS, NEG = RISK, SAFE
SEQ = ["#E8EEF6", "#C9D7EA", "#9DB6D8", "#5F86BC", "#1F4E8C", "#163A6A"]
C = {"blue": ACCENT}

RECOMMENDED = "Random Forest"
MODEL_COLOR = {
    "Logistic Regression": SLATE,
    "Decision Tree": SLATE,
    "Random Forest": ACCENT,
    "SVM": SLATE,
}
# Distinct but quiet line colours, only for the ROC chart in the collapsed section.
ROC_COLOR = {
    "Logistic Regression": "#8A94A3",
    "Decision Tree": "#C08A3E",
    "Random Forest": ACCENT,
    "SVM": "#5BA39A",
}

st.set_page_config(
    page_title="Alzheimer's Risk Screening",
    page_icon="🧠",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing models...")
def ensure_artifacts():
    """Load the committed artifacts, rebuilding them if they cannot be read.

    A pickle written by one scikit-learn version does not always load under
    another, and a hosted deployment installs its own. Rather than let that
    take the demo down, fall back to refitting from the CSV, which takes about
    twenty seconds.
    """
    try:
        models = joblib.load(ART / "models.joblib")
        ev = joblib.load(ART / "eval.joblib")
        # Touch the pipeline so a lazy incompatibility surfaces here, not mid-demo.
        # Assigned rather than left bare: Streamlit's magic renders a bare
        # expression onto the page, even inside a function.
        _probe = next(iter(models.values())).named_steps["prep"]
        assert _probe is not None
        return models, ev, None
    except Exception as exc:  # noqa: BLE001 - any load failure gets the same remedy
        import train

        train.build(search=False)
        return (
            joblib.load(ART / "models.joblib"),
            joblib.load(ART / "eval.joblib"),
            f"{type(exc).__name__}: {exc}",
        )


@st.cache_data(show_spinner=False)
def load_json(name):
    return json.loads((ART / name).read_text())


MODELS, EVAL, _rebuilt = ensure_artifacts()
SCHEMA = load_json("schema.json")
RESULTS = load_json("results.json")
DESC = load_json("descriptive.json")
CEIL = load_json("ceiling.json")
RES_BY_NAME = {r["model"]: r for r in RESULTS}


def base_layout(fig, height=340, showlegend=False):
    """Shared chart chrome: recessive grid, text-token ink, no chart junk."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=13, color=INK_2),
        margin=dict(l=8, r=16, t=24, b=8),
        hoverlabel=dict(bgcolor="#ffffff", font_size=13, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font_size=12)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font_size=12)
    return fig


def pretty(name):
    """Make an encoded feature name readable for a non-technical audience."""
    name = name.replace("_Yes", " (yes)").replace("_", ": ")
    return name.replace("Genetic Risk Factor (APOE-ε4 allele)", "APOE-ε4 allele")


def group_country(s):
    """Fold the binary-encoded Country bit columns into one readable row.

    BinaryEncoder splits Country across five bit columns (Country_0 ... Country_4)
    which mean nothing on their own. Summing them gives the feature's total
    contribution, which is what a reader actually wants to see.
    """
    bits = [i for i in s.index if i.startswith("Country_")]
    if not bits:
        return s
    out = s.drop(labels=bits)
    out["Country (all 5 encoded bits)"] = float(s[bits].sum())
    return out.sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Presentation helpers
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
      h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: {INK}; }}
      [data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
      .stTabs [data-baseweb="tab"] {{ padding: 10px 14px; font-weight: 500; color: {INK_2}; }}
      .stTabs [aria-selected="true"] {{ color: {ACCENT}; }}
      .app-eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .08em;
                      text-transform: uppercase; color: {INK_3}; margin-bottom: 2px; }}
      .app-title {{ font-size: 30px; font-weight: 700; color: {INK}; margin: 0; }}
      .app-sub {{ font-size: 15px; color: {INK_2}; margin: 4px 0 6px 0; }}
      .app-meta {{ font-size: 12.5px; color: {INK_3}; }}
      .kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px;
              padding: 14px 16px 12px 16px; height: 100%; }}
      .kpi-label {{ font-size: 12px; font-weight: 600; letter-spacing: .04em;
                    text-transform: uppercase; color: {INK_3}; }}
      .kpi-value {{ font-size: 30px; font-weight: 700; color: {INK}; line-height: 1.15;
                    margin-top: 4px; font-variant-numeric: tabular-nums; }}
      .kpi-sub {{ font-size: 13px; color: {INK_2}; margin-top: 2px; }}
      .section-title {{ font-size: 17px; font-weight: 600; color: {INK}; margin: 2px 0 2px 0; }}
      .section-sub {{ font-size: 13.5px; color: {INK_2}; margin-bottom: 4px; }}
      .takeaway {{ background: {CARD}; border: 1px solid {LINE}; border-left: 4px solid {ACCENT};
                   border-radius: 8px; padding: 12px 16px; color: {INK}; font-size: 14.5px; }}
      .takeaway b {{ color: {ACCENT}; }}
      .analyst {{ display: inline-block; font-size: 11.5px; font-weight: 600;
                  letter-spacing: .08em; text-transform: uppercase; color: {ACCENT};
                  background: {ACCENT_SOFT}; border-radius: 999px; padding: 3px 10px;
                  margin-bottom: 6px; }}
      .cm-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
      .cm-cell {{ border-radius: 10px; padding: 14px; border: 1px solid {LINE}; }}
      .cm-cell .n {{ font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }}
      .cm-cell .t {{ font-size: 13px; font-weight: 600; margin-top: 2px; }}
      .cm-cell .d {{ font-size: 12px; color: {INK_2}; margin-top: 2px; }}
      .cm-axis {{ font-size: 11.5px; color: {INK_3}; letter-spacing: .04em;
                  text-transform: uppercase; margin: 2px 0 6px 0; }}
      section[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(label, value, sub="", color=None):
    """One KPI card. Colour is reserved for values that carry risk meaning."""
    style = f' style="color:{color}"' if color else ""
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value"{style}>{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def section(title, sub=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def takeaway(text):
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)


def analyst_tag():
    st.markdown('<span class="analyst">Analyst view · behind the model</span>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar: the patient record
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Patient record")
st.sidebar.caption("Edit any field; all four models re-score immediately.")

PRESETS = {
    "Population median": {},
    "Higher-risk profile": {
        "Age": 84,
        "Genetic Risk Factor (APOE-ε4 allele)": "Yes",
        "Family History of Alzheimer's": "Yes",
        "Cognitive Test Score": 44,
        "Education Level": 4,
        "Physical Activity Level": "Low",
        "Depression Level": "High",
        "Social Engagement Level": "Low",
    },
    "Lower-risk profile": {
        "Age": 56,
        "Genetic Risk Factor (APOE-ε4 allele)": "No",
        "Family History of Alzheimer's": "No",
        "Cognitive Test Score": 92,
        "Education Level": 16,
        "Physical Activity Level": "High",
        "Depression Level": "Low",
        "Social Engagement Level": "High",
    },
}

preset_name = st.sidebar.selectbox("Start from", list(PRESETS))
preset = PRESETS[preset_name]


def default_num(col):
    return preset.get(col, SCHEMA["numeric"][col]["median"])


def default_cat(col, options):
    return options.index(preset[col]) if col in preset else 0


record = {}

with st.sidebar.expander("Demographics", expanded=True):
    record["Age"] = st.slider(
        "Age",
        int(SCHEMA["numeric"]["Age"]["min"]),
        int(SCHEMA["numeric"]["Age"]["max"]),
        int(default_num("Age")),
    )
    record["Gender"] = st.selectbox("Gender", SCHEMA["nominal"]["Gender"])
    record["Education Level"] = st.slider(
        "Education (years)",
        int(SCHEMA["numeric"]["Education Level"]["min"]),
        int(SCHEMA["numeric"]["Education Level"]["max"]),
        int(default_num("Education Level")),
    )
    record["Country"] = st.selectbox("Country", SCHEMA["high_card"]["Country"])
    record["Urban vs Rural Living"] = st.selectbox(
        "Living area", SCHEMA["nominal"]["Urban vs Rural Living"]
    )
    record["Marital Status"] = st.selectbox(
        "Marital status", SCHEMA["nominal"]["Marital Status"]
    )
    record["Employment Status"] = st.selectbox(
        "Employment", SCHEMA["nominal"]["Employment Status"]
    )

with st.sidebar.expander("Genetic and clinical", expanded=True):
    opts = SCHEMA["nominal"]["Genetic Risk Factor (APOE-ε4 allele)"]
    record["Genetic Risk Factor (APOE-ε4 allele)"] = st.selectbox(
        "APOE-ε4 allele",
        opts,
        index=default_cat("Genetic Risk Factor (APOE-ε4 allele)", opts),
    )
    opts = SCHEMA["nominal"]["Family History of Alzheimer's"]
    record["Family History of Alzheimer's"] = st.selectbox(
        "Family history", opts, index=default_cat("Family History of Alzheimer's", opts)
    )
    record["Cognitive Test Score"] = st.slider(
        "Cognitive test score",
        int(SCHEMA["numeric"]["Cognitive Test Score"]["min"]),
        int(SCHEMA["numeric"]["Cognitive Test Score"]["max"]),
        int(default_num("Cognitive Test Score")),
    )
    record["BMI"] = st.slider(
        "BMI",
        float(SCHEMA["numeric"]["BMI"]["min"]),
        float(SCHEMA["numeric"]["BMI"]["max"]),
        float(default_num("BMI")),
        step=0.1,
    )
    record["Diabetes"] = st.selectbox("Diabetes", SCHEMA["nominal"]["Diabetes"])
    record["Hypertension"] = st.selectbox("Hypertension", SCHEMA["nominal"]["Hypertension"])
    record["Cholesterol Level"] = st.selectbox(
        "Cholesterol", SCHEMA["nominal"]["Cholesterol Level"]
    )

with st.sidebar.expander("Lifestyle", expanded=False):
    for col in [
        "Physical Activity Level",
        "Depression Level",
        "Social Engagement Level",
        "Stress Levels",
        "Air Pollution Exposure",
        "Income Level",
    ]:
        o = SCHEMA["ordinal"][col]
        record[col] = st.selectbox(col, o, index=default_cat(col, o))
    for col in ["Smoking Status", "Alcohol Consumption", "Sleep Quality", "Dietary Habits"]:
        record[col] = st.selectbox(col, SCHEMA["nominal"][col])

patient = pd.DataFrame([[record[c] for c in SCHEMA["column_order"]]], columns=SCHEMA["column_order"])


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-eyebrow">Risk screening</div>
    <div class="app-title">Alzheimer's Risk Screening</div>
    <div class="app-sub">Score a patient's risk, then set the referral cut-off the business can live with.</div>
    <div class="app-meta">{DESC['n_rows']:,} records &nbsp;·&nbsp; 4 models compared &nbsp;·&nbsp;
    evaluated on a held-out 20% test set</div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

tab_pred, tab_pres, tab_diag, tab_ceil = st.tabs(
    ["Score a patient", "Set the cut-off", "What drives it", "Where it tops out"]
)


# ----------------------------------------------------------------------------
# SCORE A PATIENT
# ----------------------------------------------------------------------------
with tab_pred:
    scores = {name: float(m.predict_proba(patient)[0, 1]) for name, m in MODELS.items()}
    rec_score = scores[RECOMMENDED]
    n_flagged = sum(v >= 0.5 for v in scores.values())
    spread = max(scores.values()) - min(scores.values())
    agreement = "Strong" if spread < 0.10 else ("Moderate" if spread < 0.20 else "Weak")

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi(
            "Risk score",
            f"{rec_score:.0%}",
            ("Above" if rec_score >= 0.5 else "Below") + f" the 50% cut-off · {RECOMMENDED}",
            color=RISK if rec_score >= 0.5 else SAFE,
        )
    with k2:
        kpi("Models flagging", f"{n_flagged} of 4", "at the default 50% cut-off")
    with k3:
        kpi("Model agreement", agreement, f"{spread:.0%} gap between highest and lowest score")

    st.markdown("")
    with st.container(border=True):
        section("Risk score by model", "Each model scores the same record. Navy is the recommended model.")
        order = sorted(scores, key=scores.get, reverse=True)
        fig = go.Figure()
        for name in order:
            fig.add_trace(
                go.Bar(
                    x=[scores[name]],
                    y=[name],
                    orientation="h",
                    marker=dict(color=MODEL_COLOR[name]),
                    width=0.56,
                    text=[f"{scores[name]:.0%}"],
                    textposition="outside",
                    textfont=dict(color=INK, size=13),
                    hovertemplate=f"{name}<br>%{{x:.1%}}<extra></extra>",
                )
            )
        fig.add_vline(x=0.5, line=dict(color=INK_3, width=1.5, dash="dot"))
        fig.add_annotation(x=0.5, y=1.08, yref="paper", text="50% cut-off", showarrow=False,
                           font=dict(size=11, color=INK_3))
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=False)
        fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
        st.plotly_chart(base_layout(fig, height=230), width="stretch")

    with st.container(border=True):
        section(
            "Cases caught (recall)",
            "Share of true cases each model catches on the test set, at the default 50% cut-off.",
        )
        rec = sorted(RESULTS, key=lambda r: r["recall"], reverse=True)
        fig = go.Figure()
        for r in rec:
            label = r["model"] + ("  · recommended" if r["model"] == RECOMMENDED else "")
            fig.add_trace(
                go.Bar(
                    x=[r["recall"]],
                    y=[label],
                    orientation="h",
                    marker=dict(color=MODEL_COLOR[r["model"]]),
                    width=0.56,
                    text=[f"{r['recall']:.1%}"],
                    textposition="outside",
                    textfont=dict(color=INK, size=13),
                    hovertemplate=f"{r['model']}<br>%{{x:.1%}} of true cases caught<extra></extra>",
                )
            )
        labels = [r["model"] + ("  · recommended" if r["model"] == RECOMMENDED else "") for r in rec]
        fig.update_xaxes(range=[0, 1.1], tickformat=".0%", showgrid=False)
        fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
        st.plotly_chart(base_layout(fig, height=230), width="stretch")
        st.caption(
            "Random Forest is recommended: it ranks risk best (AUC 0.803) and gives smooth "
            "scores, so the cut-off can be tuned precisely. The decision tree's lead here "
            "comes from its few coarse score levels; at equal recall it refers more people."
        )

    with st.expander("Full evaluation (accuracy, precision, F1, AUC, ROC curves)"):
        perf = pd.DataFrame(
            [
                {
                    "Model": r["model"],
                    "Accuracy": r["accuracy"],
                    "Recall": r["recall"],
                    "Precision": r["precision"],
                    "F1": r["f1"],
                    "AUC": r["roc_auc"],
                    "Missed cases": r["confusion"]["fn"],
                    "Notebook acc.": r["notebook_accuracy"],
                }
                for r in RESULTS
            ]
        ).sort_values("Recall", ascending=False)

        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.dataframe(
                perf.style.format(
                    {
                        "Accuracy": "{:.4f}",
                        "Recall": "{:.4f}",
                        "Precision": "{:.4f}",
                        "F1": "{:.4f}",
                        "AUC": "{:.4f}",
                        "Notebook acc.": "{:.4f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Accuracy reproduces the original notebook within 0.004 after moving every "
                "encoder inside the cross-validation pipeline, which confirms the earlier "
                "scaling leakage was real but immaterial."
            )
        with c2:
            fig = go.Figure()
            for r in RESULTS:
                fig.add_trace(
                    go.Scatter(
                        x=r["roc"]["fpr"],
                        y=r["roc"]["tpr"],
                        mode="lines",
                        name=f"{r['model']} ({r['roc_auc']:.3f})",
                        line=dict(color=ROC_COLOR[r["model"]], width=2.5 if r["model"] == RECOMMENDED else 1.8),
                        hovertemplate=f"{r['model']}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>",
                    )
                )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                           line=dict(color=GRID, width=2, dash="dash"),
                           hoverinfo="skip", showlegend=False)
            )
            fig.update_xaxes(title="False positive rate", range=[0, 1])
            fig.update_yaxes(title="True positive rate", range=[0, 1])
            st.plotly_chart(base_layout(fig, height=340, showlegend=True), width="stretch")
            st.caption(
                "The four curves sit almost on top of each other (AUC 0.789 to 0.803): "
                "the ceiling is the feature set, not the algorithm."
            )


# ----------------------------------------------------------------------------
# SET THE CUT-OFF
# ----------------------------------------------------------------------------
with tab_pres:
    y_true = EVAL["y_test"]

    with st.container(border=True):
        section(
            "Business inputs",
            "How much worse is a missed case than a false alarm? That ratio is a business "
            "decision; the cut-off follows from it.",
        )
        c1, c2, c3 = st.columns(3)
        model_name = c1.selectbox("Model", list(MODELS), index=list(MODELS).index(RECOMMENDED))
        fn_cost = c2.number_input("Cost of a missed case", 1, 500, 5, step=1)
        fp_cost = c3.number_input("Cost of a false alarm", 1, 500, 1, step=1)

        proba = EVAL["probas"][model_name]
        grid = np.linspace(0.05, 0.95, 91)
        tp = np.array([int(((proba >= t) & (y_true == 1)).sum()) for t in grid])
        fp = np.array([int(((proba >= t) & (y_true == 0)).sum()) for t in grid])
        fn = np.array([int(((proba < t) & (y_true == 1)).sum()) for t in grid])
        tn = np.array([int(((proba < t) & (y_true == 0)).sum()) for t in grid])
        cost = fn * fn_cost + fp * fp_cost
        best_t = float(grid[int(np.argmin(cost))])

        threshold = st.slider(
            "Referral cut-off (risk score at or above this is referred)",
            0.05, 0.95, round(best_t, 2), step=0.01,
            help="Starts at the lowest-cost cut-off for the ratio above. Drag to explore.",
        )

    i = int(np.abs(grid - threshold).argmin())
    positives = tp[i] + fn[i]
    recall_at = tp[i] / max(1, positives)
    precision_at = tp[i] / max(1, tp[i] + fp[i])
    referred = tp[i] + fp[i]
    share = referred / len(y_true)

    st.markdown("")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Cases caught", f"{recall_at:.0%}", f"{tp[i]:,} of {positives:,} true cases · recall")
    with k2:
        kpi("Cases missed", f"{fn[i]:,}", "true cases sent home", color=RISK)
    with k3:
        kpi("People referred", f"{share:.0%}", f"{referred:,} of {len(y_true):,} screened")
    with k4:
        kpi("Referred who are at risk", f"{precision_at:.0%}",
            f"{tp[i]:,} of {referred:,} · precision")

    st.markdown("")
    left, right = st.columns([1.45, 1])

    with left:
        with st.container(border=True):
            section("The tradeoff", "Lowering the cut-off catches more cases but adds false alarms.")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=grid, y=fn, mode="lines", name="Cases missed",
                    line=dict(color=RISK, width=2.5),
                    hovertemplate="Cut-off %{x:.2f}<br>Missed %{y:,}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=grid, y=fp, mode="lines", name="False alarms",
                    line=dict(color=ACCENT, width=2.5),
                    hovertemplate="Cut-off %{x:.2f}<br>False alarms %{y:,}<extra></extra>",
                )
            )
            fig.add_vline(x=threshold, line=dict(color=INK, width=1.5))
            if abs(threshold - best_t) > 0.005:
                fig.add_vline(x=best_t, line=dict(color=INK_3, width=1.5, dash="dot"))
            fig.add_annotation(
                x=best_t, y=0.96, yref="paper", text=f"lowest cost · {best_t:.2f}",
                showarrow=False, xanchor="left", xshift=6, font=dict(size=11, color=INK_2),
            )
            fig.update_xaxes(title="Referral cut-off")
            fig.update_yaxes(title="People in the test set")
            st.plotly_chart(base_layout(fig, height=330, showlegend=True), width="stretch")

    with right:
        with st.container(border=True):
            section("Outcomes at this cut-off", f"{model_name} · cut-off {threshold:.2f}")

            def cell(n, title, desc, fg, bg):
                return (f'<div class="cm-cell" style="background:{bg}">'
                        f'<div class="n" style="color:{fg}">{n:,}</div>'
                        f'<div class="t" style="color:{fg}">{title}</div>'
                        f'<div class="d">{desc}</div></div>')

            st.markdown(
                '<div class="cm-axis">Has it</div><div class="cm-grid">'
                + cell(tp[i], "Caught", "Referred, and at risk", SAFE, "#EAF5F2")
                + cell(fn[i], "Missed", "Sent home, but at risk", RISK, "#FBEDEB")
                + '</div><div class="cm-axis" style="margin-top:12px">Healthy</div><div class="cm-grid">'
                + cell(fp[i], "False alarm", "Referred, but healthy", ACCENT, "#EDF2F8")
                + cell(tn[i], "Correctly cleared", "Sent home, healthy", INK_2, "#F3F5F8")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    if share > 0.5:
        takeaway(
            f"<b>So what:</b> at a {fn_cost}:{fp_cost} cost ratio, catching {recall_at:.0%} of "
            f"cases means referring {share:.0%} of everyone, and about "
            f"{1 - precision_at:.0%} of those referrals are false alarms. The model still "
            "beats referring everyone, but real triage needs better data, not a better model."
        )
    else:
        takeaway(
            f"<b>So what:</b> at a {fn_cost}:{fp_cost} cost ratio, the lowest-cost cut-off is "
            f"{best_t:.2f}, not 0.50. It catches {recall_at:.0%} of cases while referring "
            f"{share:.0%} of people."
        )


# ----------------------------------------------------------------------------
# DIAGNOSTIC
# ----------------------------------------------------------------------------
with tab_diag:
    analyst_tag()
    st.markdown("#### What is actually driving the prediction?")

    imp = group_country(EVAL["importances"]).head(12)[::-1]
    mi = group_country(EVAL["mutual_info"]).head(12)[::-1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Random Forest feature importance**")
        fig = go.Figure(
            go.Bar(
                x=imp.values,
                y=[pretty(n) for n in imp.index],
                orientation="h",
                marker=dict(color=SEQ[4]),
                width=0.62,
                text=[f"{v:.3f}" for v in imp.values],
                textposition="outside",
                textfont=dict(color=INK_2, size=12),
                hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Importance", range=[0, float(imp.max()) * 1.2])
        st.plotly_chart(base_layout(fig, height=420), width='stretch')
    with c2:
        st.markdown("**Mutual information with diagnosis**")
        fig = go.Figure(
            go.Bar(
                x=mi.values,
                y=[pretty(n) for n in mi.index],
                orientation="h",
                marker=dict(color=SEQ[3]),
                width=0.62,
                text=[f"{v:.3f}" for v in mi.values],
                textposition="outside",
                textfont=dict(color=INK_2, size=12),
                hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Mutual information", range=[0, float(mi.max()) * 1.2])
        st.plotly_chart(base_layout(fig, height=420), width='stretch')

    st.caption(
        "Country is binary-encoded across five bit columns, which mean nothing "
        "individually, so both charts show their summed contribution as a single row."
    )

    takeaway(
        "<b>Takeaway:</b> two independent methods agree on the same three features: age, the APOE-ε4 "
        "allele, and family history. That matches the established clinical literature, "
        "which is the reassuring result. The flip side is that the remaining twenty "
        "lifestyle and demographic features carry almost no signal, which is why all "
        "four models plateau near 73% and why biomarker features would be the real "
        "next step."
    )

    st.divider()
    st.markdown("#### Positive rate by category")
    cats = list(DESC["positive_rate_by_category"])
    pick = st.selectbox("Feature", cats, index=cats.index("Genetic Risk Factor (APOE-ε4 allele)"))
    rates = DESC["positive_rate_by_category"][pick]
    base = DESC["target_balance"]["Yes"]

    fig = go.Figure(
        go.Bar(
            x=list(rates),
            y=list(rates.values()),
            marker=dict(color=[POS if v > base else C["blue"] for v in rates.values()]),
            width=0.5,
            text=[f"{v:.1%}" for v in rates.values()],
            textposition="outside",
            textfont=dict(color=INK_2, size=13),
            hovertemplate="%{x}<br>%{y:.2%} positive<extra></extra>",
        )
    )
    fig.add_hline(
        y=base, line=dict(color=INK_2, width=2, dash="dot"),
        annotation_text=f"population rate {base:.1%}", annotation_position="top left",
        annotation_font=dict(size=11, color=INK_2),
    )
    fig.update_yaxes(title="Share diagnosed", tickformat=".0%", range=[0, max(rates.values()) * 1.25])
    st.plotly_chart(base_layout(fig, height=320), width='stretch')


# ----------------------------------------------------------------------------
# CEILING
# ----------------------------------------------------------------------------
with tab_ceil:
    analyst_tag()
    st.markdown("#### How much is the modelling actually buying?")
    st.caption(
        "Before spending another week tuning, it is worth testing whether the limit "
        "is the algorithm or the data. Three cheaper models and a learning curve "
        "answer that."
    )

    naive = CEIL["naive_baseline"]
    age_only = next(m for m in CEIL["models"] if m["label"] == "Age alone")
    best = CEIL["best_tuned"]
    lift = best["accuracy"] - age_only["accuracy"]

    m1, m2, m3 = st.columns(3)
    with m1:
        kpi("Always answer No", f"{naive:.1%}", "majority class, the bar to beat")
    with m2:
        kpi("Age alone", f"{age_only['accuracy']:.1%}", "one variable")
    with m3:
        kpi(f"Best tuned · {best['label']}", f"{best['accuracy']:.1%}",
            f"{lift:+.1%} over age alone")

    st.markdown("")
    bars = (
        [{"label": "Always answer No", "accuracy": naive, "n_features": 0,
          "note": "The majority class. Every result has to beat this."}]
        + CEIL["models"]
        + [{"label": f"Best tuned ({best['label']})", "accuracy": best["accuracy"],
            "n_features": 24, "note": "All 24 features, four algorithms, full grid search."}]
    )

    left, right = st.columns([1.3, 1])

    with left:
        ys = [b["label"] for b in bars][::-1]
        xs = [b["accuracy"] for b in bars][::-1]
        fig = go.Figure(
            go.Bar(
                x=xs,
                y=ys,
                orientation="h",
                marker=dict(color=[SEQ[1] if v == naive else SEQ[4] for v in xs]),
                width=0.58,
                text=[f"{v:.1%}" for v in xs],
                textposition="outside",
                textfont=dict(color=INK_2, size=13),
                hovertemplate="%{y}<br>%{x:.2%}<extra></extra>",
            )
        )
        fig.add_vline(x=naive, line=dict(color=INK_2, width=2, dash="dot"))
        fig.update_xaxes(title="Test accuracy", range=[0, 0.88], tickformat=".0%")
        st.plotly_chart(base_layout(fig, height=330), width="stretch")
        st.caption(
            f"Dotted line is the naive baseline. One variable gets {age_only['accuracy']:.1%}. "
            f"Twenty-three more features, four algorithms and a full grid search add "
            f"{lift:.1%}."
        )

    with right:
        curve = CEIL["learning_curve"]
        fig = go.Figure(
            go.Scatter(
                x=[c["n"] for c in curve],
                y=[c["accuracy"] for c in curve],
                mode="lines+markers",
                line=dict(color=C["blue"], width=2),
                marker=dict(size=9, color=C["blue"], line=dict(width=2, color=SURFACE)),
                hovertemplate="%{x:,} training rows<br>%{y:.2%} CV accuracy<extra></extra>",
            )
        )
        fig.add_hline(
            y=naive,
            line=dict(color=INK_2, width=2, dash="dot"),
            annotation_text="always answer No",
            annotation_position="bottom left",
            annotation_font=dict(size=11, color=INK_2),
        )
        fig.update_xaxes(title="Training rows")
        # Anchored to a full, honest range. Letting Plotly auto-scale zooms into a
        # one-point window and turns a flat line into a dramatic climb.
        fig.update_yaxes(
            title="Cross-validated accuracy", tickformat=".0%", range=[0.5, 0.85]
        )
        st.plotly_chart(base_layout(fig, height=330), width="stretch")
        first, last = curve[0], curve[-1]
        st.caption(
            f"Ten times more data ({first['n']:,} to {last['n']:,} rows) moves accuracy "
            f"by {last['accuracy'] - first['accuracy']:.1%}, and the curve is flattening. "
            "Collecting more of the same data will not help either."
        )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What this rules out**")
        st.markdown(
            "- **A better algorithm.** Gradient boosting, which usually wins on tabular "
            "data of this size, scored "
            f"{next(m for m in CEIL['models'] if m['label'] == 'Gradient boosting')['accuracy']:.1%}, "
            "no better than the tuned decision tree.\n"
            "- **More rows.** The learning curve has gone flat.\n"
            "- **More tuning.** The four tuned models land within 1.8 points of each other."
        )
    with c2:
        st.markdown("**What would actually move it**")
        st.markdown(
            "- **Biomarkers.** CSF amyloid and tau, or MRI hippocampal volume. These "
            "separate cases in the clinical literature in a way demographics cannot.\n"
            "- **Longitudinal records.** Change in cognitive score over time carries far "
            "more signal than a single reading.\n"
            "- **A real cohort.** The age pattern below suggests this data is at least "
            "partly synthetic, which caps what any model can learn from it."
        )

    st.divider()
    st.markdown("**The evidence for that last point**")

    bins = pd.DataFrame(DESC["age_bins"])
    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig = go.Figure(
            go.Scatter(
                x=bins["_b"],
                y=bins["mean"],
                mode="lines+markers",
                line=dict(color=C["blue"], width=2),
                marker=dict(size=9, color=C["blue"], line=dict(width=2, color=SURFACE)),
                hovertemplate="Age %{x}<br>%{y:.1%} diagnosed<br>%{customdata:,} patients<extra></extra>",
                customdata=bins["count"],
            )
        )
        fig.add_hline(
            y=DESC["target_balance"]["Yes"],
            line=dict(color=INK_2, width=2, dash="dot"),
            annotation_text="population rate",
            annotation_position="top left",
            annotation_font=dict(size=11, color=INK_2),
        )
        fig.update_xaxes(title="Age band")
        fig.update_yaxes(title="Share diagnosed", tickformat=".0%")
        st.plotly_chart(base_layout(fig, height=320), width="stretch")
    with c2:
        st.markdown("")
        st.markdown(
            "Diagnosis rate holds flat to 65, jumps, holds flat again, jumps, then "
            "plateaus after 75.\n\n"
            "Real incidence climbs smoothly and keeps climbing into the eighties and "
            "nineties. A step function like this is what generated data looks like, "
            "not a cohort.\n\n"
            "It does not invalidate the modelling, but it does cap how far any "
            "conclusion here should travel, and it is worth saying before someone "
            "else notices."
        )

    takeaway(
        "<b>Takeaway:</b> the point of this panel is knowing when to stop. The modelling here is sound "
        "and the ceiling is the feature set, so the next investment belongs in data "
        "collection, not in a fifth algorithm. On a site problem the same test is worth "
        "running early, because it decides whether to spend the next sprint on the model "
        "or on the instrumentation feeding it."
    )
