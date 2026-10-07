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

# Categorical palette, slots 1-4, validated for the light chart surface.
# Light mode carries a contrast WARN on aqua/yellow, so every chart using
# those slots also ships direct labels or an adjacent table.
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"]]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e8e7e3"
SURFACE = "#fcfcfb"
POS = "#e34948"  # positive / at-risk
NEG = "#2a78d6"  # negative / not at risk

MODEL_COLOR = {
    "Logistic Regression": C["blue"],
    "Decision Tree": C["orange"],
    "Random Forest": C["aqua"],
    "SVM": C["yellow"],
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
        font=dict(family="system-ui, -apple-system, sans-serif", size=13, color=INK_2),
        margin=dict(l=8, r=8, t=28, b=8),
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
st.markdown("## Alzheimer's Risk Screening")
st.caption(
    f"{DESC['n_rows']:,} records · {DESC['n_features']} features · "
    "four classifiers compared · held-out 20% test split"
)

tab_pred, tab_pres, tab_diag, tab_ceil = st.tabs(
    ["Score a patient", "Set the cut-off", "What drives it", "Where it tops out"]
)


# ----------------------------------------------------------------------------
# PREDICTIVE
# ----------------------------------------------------------------------------
with tab_pred:
    st.markdown("#### Risk score for this patient")

    scores = {name: float(m.predict_proba(patient)[0, 1]) for name, m in MODELS.items()}
    mean_score = float(np.mean(list(scores.values())))
    n_flagged = sum(v >= 0.5 for v in scores.values())

    left, right = st.columns([1, 1.6])

    with left:
        # Hero number: a single headline figure reads better as type than as a
        # gauge, which asks the viewer to judge an angle.
        tone = POS if mean_score >= 0.5 else NEG
        verdict = "Above the 0.50 cut-off" if mean_score >= 0.5 else "Below the 0.50 cut-off"
        st.markdown(
            f"""
            <div style="padding:4px 0 2px 0">
              <div style="font-size:13px;color:{INK_2};letter-spacing:.02em">
                MEAN RISK SCORE
              </div>
              <div style="font-size:68px;line-height:1.05;font-weight:600;color:{tone}">
                {mean_score:.1%}
              </div>
              <div style="font-size:14px;color:{INK_2};margin-top:2px">{verdict}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"{n_flagged} of 4 models flag this patient at the default 0.50 cut-off. "
            "Mean of the four model probabilities; the individual scores are beside it."
        )
        spread = max(scores.values()) - min(scores.values())
        st.metric("Spread across models", f"{spread:.1%}")
        st.caption(
            "A wide spread means the models disagree, which is itself worth surfacing "
            "to a reviewer."
        )

    with right:
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
                    text=[f"{scores[name]:.1%}"],
                    textposition="outside",
                    textfont=dict(color=INK_2, size=13),
                    hovertemplate=f"{name}<br>%{{x:.1%}}<extra></extra>",
                )
            )
        fig.add_vline(x=0.5, line=dict(color=INK_2, width=2, dash="dot"))
        fig.update_xaxes(range=[0, 1.14], tickformat=".0%", title="Predicted probability")
        fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(base_layout(fig, height=260), width='stretch')
        st.caption("Dotted line is the default 0.50 decision threshold.")

    st.divider()
    st.markdown("#### Held-out test performance")

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
            ).background_gradient(subset=["Recall"], cmap="Blues"),
            hide_index=True,
            width='stretch',
        )
        st.caption(
            "Sorted by recall, not accuracy, because recall is what a screening tool is "
            "really judged on. Accuracy reproduces the original notebook within 0.004 "
            "after moving every encoder inside the cross-validation pipeline, which "
            "confirms the earlier scaling leakage was real but immaterial."
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
                    line=dict(color=MODEL_COLOR[r["model"]], width=2),
                    hovertemplate=f"{r['model']}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=GRID, width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1])
        st.plotly_chart(base_layout(fig, height=340, showlegend=True), width='stretch')
        st.caption(
            "The four curves sit almost on top of each other (AUC 0.789 to 0.803). "
            "That overlap is the finding: the ceiling here is the feature set, not the "
            "choice of algorithm."
        )


# ----------------------------------------------------------------------------
# PRESCRIPTIVE
# ----------------------------------------------------------------------------
with tab_pres:
    st.markdown("#### Where should the screening threshold sit?")
    st.caption(
        "A screening tool is not scored on accuracy. Missing a true case costs far more "
        "than a false alarm, so the operating point belongs to the clinic, not the default."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    model_name = c1.selectbox("Model", list(MODELS), index=list(MODELS).index("Random Forest"))
    fn_cost = c2.number_input("Cost of a missed case", 1, 500, 5, step=1)
    fp_cost = c3.number_input("Cost of a false alarm", 1, 500, 1, step=1)

    y_true = EVAL["y_test"]
    proba = EVAL["probas"][model_name]

    grid = np.linspace(0.05, 0.95, 91)
    tp = np.array([int(((proba >= t) & (y_true == 1)).sum()) for t in grid])
    fp = np.array([int(((proba >= t) & (y_true == 0)).sum()) for t in grid])
    fn = np.array([int(((proba < t) & (y_true == 1)).sum()) for t in grid])
    tn = np.array([int(((proba < t) & (y_true == 0)).sum()) for t in grid])
    cost = fn * fn_cost + fp * fp_cost
    best_t = float(grid[int(np.argmin(cost))])

    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, round(best_t, 2), step=0.01,
        help="Patients at or above this probability are referred for follow-up.",
    )

    i = int(np.abs(grid - threshold).argmin())
    recall_at = tp[i] / max(1, tp[i] + fn[i])
    precision_at = tp[i] / max(1, tp[i] + fp[i])
    referred = tp[i] + fp[i]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cases caught", f"{recall_at:.1%}", f"{tp[i]:,} of {tp[i] + fn[i]:,}")
    m2.metric(
        "Cases missed",
        f"{fn[i]:,}",
        f"{fn[i] - fn[int(np.abs(grid - 0.5).argmin())]:+,} vs 0.50",
        delta_color="inverse",  # fewer missed cases is an improvement
    )
    m3.metric("Referred for follow-up", f"{referred:,}", f"{referred / len(y_true):.0%} of population")
    m4.metric("Precision of referrals", f"{precision_at:.1%}")

    left, right = st.columns([1.5, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid, y=fn, mode="lines", name="Missed cases (FN)",
                line=dict(color=POS, width=2),
                hovertemplate="Threshold %{x:.2f}<br>Missed %{y:,}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid, y=fp, mode="lines", name="False alarms (FP)",
                line=dict(color=C["blue"], width=2),
                hovertemplate="Threshold %{x:.2f}<br>False alarms %{y:,}<extra></extra>",
            )
        )
        fig.add_vline(x=threshold, line=dict(color=INK_2, width=2))
        fig.add_vline(x=best_t, line=dict(color=INK_2, width=2, dash="dot"))
        fig.add_annotation(
            x=best_t, y=0.94, yref="paper", text=f"cost-minimising {best_t:.2f}",
            showarrow=False, xanchor="left", xshift=6,
            font=dict(size=11, color=INK_2),
        )
        fig.update_xaxes(title="Decision threshold")
        fig.update_yaxes(title="Patients in the test split")
        st.plotly_chart(base_layout(fig, height=360, showlegend=True), width='stretch')

        share = referred / len(y_true)
        note = (
            f"At the cost ratio {fn_cost}:{fp_cost}, total cost is minimised at "
            f"{best_t:.2f}, not 0.50."
        )
        if share > 0.5:
            note += (
                f"  Note that this refers {share:.0%} of the population, which no clinic "
                "could absorb. Push the cost of a missed case high enough and the maths "
                "says screen almost everyone, which is the honest signal that this "
                "feature set cannot separate the classes sharply enough to triage on."
            )
        st.caption(note)

    with right:
        cm = np.array([[tn[i], fp[i]], [fn[i], tp[i]]])
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                colorscale=[[0, SEQ[0]], [1, SEQ[4]]],
                showscale=False,
                text=[[f"{v:,}" for v in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(size=17, color=INK),
                hovertemplate="%{y} · %{x}<br>%{z:,}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(base_layout(fig, height=360), width='stretch')
        st.caption(f"Confusion matrix for {model_name} at threshold {threshold:.2f}.")


# ----------------------------------------------------------------------------
# DIAGNOSTIC
# ----------------------------------------------------------------------------
with tab_diag:
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

    st.info(
        "Two independent methods agree on the same three features: age, the APOE-ε4 "
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
    m1.metric("Always answer No", f"{naive:.1%}", help="Predict the majority class every time.")
    m2.metric("Age alone, one variable", f"{age_only['accuracy']:.1%}")
    m3.metric(
        f"Best tuned model ({best['label']})",
        f"{best['accuracy']:.1%}",
        f"{lift:+.1%} over age alone",
    )

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

    st.info(
        "The point of this panel is knowing when to stop. The modelling here is sound "
        "and the ceiling is the feature set, so the next investment belongs in data "
        "collection, not in a fifth algorithm. On a site problem the same test is worth "
        "running early, because it decides whether to spend the next sprint on the model "
        "or on the instrumentation feeding it."
    )
