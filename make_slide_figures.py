"""Render the descriptive-analytics figures as standalone PNGs for a deck.

These no longer live in the app, since a deployed screening tool would not ship
an EDA panel. They are still worth having on a slide.

Run:  python make_slide_figures.py    (writes slide_figures/)
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
OUT = ROOT / "slide_figures"
OUT.mkdir(exist_ok=True)

BLUE, POS, INK, INK_2 = "#2a78d6", "#e34948", "#0b0b0b", "#52514e"
GRID, SURFACE = "#e8e7e3", "#ffffff"

DESC = json.loads((ART / "descriptive.json").read_text())


def chrome(fig, title, height=560):
    fig.update_layout(
        title=dict(text=title, font=dict(size=21, color=INK), x=0, xanchor="left"),
        height=height,
        width=1000,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif", size=15, color=INK_2),
        margin=dict(l=70, r=40, t=70, b=60),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# 1. Class balance against the naive baseline.
bal = DESC["target_balance"]
fig = go.Figure(
    go.Bar(
        x=[bal["No"], bal["Yes"]],
        y=["Not diagnosed", "Diagnosed"],
        orientation="h",
        marker=dict(color=[BLUE, POS]),
        width=0.5,
        text=[f"{bal['No']:.1%}", f"{bal['Yes']:.1%}"],
        textposition="outside",
        textfont=dict(size=16, color=INK_2),
    )
)
fig.update_xaxes(title="Share of 74,283 records", range=[0, 0.72], tickformat=".0%")
chrome(fig, "Class balance: a model answering No every time scores 58.7%", 400)
fig.write_image(OUT / "01_class_balance.png", scale=2)

# 2. Diagnosis rate by age band.
bins = pd.DataFrame(DESC["age_bins"])
fig = go.Figure(
    go.Scatter(
        x=bins["_b"],
        y=bins["mean"],
        mode="lines+markers",
        line=dict(color=BLUE, width=3),
        marker=dict(size=12, color=BLUE, line=dict(width=2, color=SURFACE)),
    )
)
fig.add_hline(
    y=bal["Yes"],
    line=dict(color=INK_2, width=2, dash="dot"),
    annotation_text="population rate 41.3%",
    annotation_position="top left",
    annotation_font=dict(size=13, color=INK_2),
)
fig.update_xaxes(title="Age band")
fig.update_yaxes(title="Share diagnosed", tickformat=".0%")
chrome(fig, "Diagnosis rate rises in flat steps, then plateaus after 75")
fig.write_image(OUT / "02_age_pattern.png", scale=2)

# 3. Numeric separation between the two classes.
rows = []
for col, stats in DESC["numeric_by_class"].items():
    rows.append(
        {
            "feature": col,
            "Not diagnosed": stats["No"]["mean"],
            "Diagnosed": stats["Yes"]["mean"],
            "gap": stats["Yes"]["mean"] - stats["No"]["mean"],
        }
    )
d = pd.DataFrame(rows).sort_values("gap", key=abs)
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=d["Not diagnosed"], y=d["feature"], orientation="h", name="Not diagnosed",
        marker=dict(color=BLUE), width=0.34,
        text=[f"{v:.1f}" for v in d["Not diagnosed"]], textposition="outside",
        textfont=dict(size=13, color=INK_2),
    )
)
fig.add_trace(
    go.Bar(
        x=d["Diagnosed"], y=d["feature"], orientation="h", name="Diagnosed",
        marker=dict(color=POS), width=0.34,
        text=[f"{v:.1f}" for v in d["Diagnosed"]], textposition="outside",
        textfont=dict(size=13, color=INK_2),
    )
)
fig.update_xaxes(title="Group mean", range=[0, 95])
chrome(fig, "Only age separates the two groups meaningfully", 520)
# Legend sits above the plot area, clear of the top bar, so the extra top
# margin is applied after chrome() rather than fighting it.
fig.update_layout(
    barmode="group",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
    margin=dict(l=160, r=60, t=120, b=70),
)
fig.write_image(OUT / "03_numeric_separation.png", scale=2)

print("wrote:")
for f in sorted(OUT.iterdir()):
    print(" ", f.name)
