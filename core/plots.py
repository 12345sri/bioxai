"""Charts used across the app (matplotlib, consistent palette)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

INK = "#1E1B2E"
ROSE = "#B0306A"
TEAL = "#2A7F8F"
SAND = "#C9A227"
MUTE = "#9A93A6"
PALE = "#F4EEF2"
SERIES = [ROSE, TEAL, SAND, "#6B4E9B", MUTE]

plt.rcParams.update({
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "font.size": 10, "figure.dpi": 110,
})


def _fig(w=6.4, h=4.2):
    return plt.subplots(figsize=(w, h))


# ------------------------------------------------------------------ genomics
def class_balance(n_before, n_after, labels):
    fig, ax = _fig(5.5, 3.2)
    x = np.arange(2)
    ax.bar(x - 0.2, n_before, 0.4, color=MUTE, label="Training set (original)")
    ax.bar(x + 0.2, n_after, 0.4, color=ROSE, label="After SMOTE")
    ax.set_xticks(x, [labels[0], labels[1]])
    ax.set_ylabel("Samples")
    ax.legend(frameon=False)
    ax.set_title("Class balance in the training data")
    fig.tight_layout()
    return fig


def roc_curves(rocs: dict):
    fig, ax = _fig(5.6, 5.0)
    for (name, (fpr, tpr, auc)), c in zip(rocs.items(), SERIES):
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1, label="Chance (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    return fig


def confusion(cm, labels):
    fig, ax = _fig(3.8, 3.3)
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(ROSE, as_cmap=True), cbar=False,
                xticklabels=[f"Pred. {labels[0]}", f"Pred. {labels[1]}"],
                yticklabels=[f"True {labels[0]}", f"True {labels[1]}"], ax=ax)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def importance_bar(imp: pd.DataFrame, top: int = 15):
    d = imp.head(top).iloc[::-1]
    colors = [ROSE if v > 0.1 else (TEAL if v < -0.1 else MUTE) for v in d["direction"]]
    fig, ax = _fig(6.2, 0.32 * len(d) + 1.2)
    ax.barh(d["gene"], d["importance"], color=colors)
    ax.set_xlabel(imp.attrs.get("method", "Importance"))
    ax.set_title(f"Top genes driving the {imp.attrs.get('model', '')} predictions")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ROSE, label="Higher → late stage"),
                       Patch(color=TEAL, label="Higher → early stage"),
                       Patch(color=MUTE, label="Mixed")], frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def expression_heatmap(X: pd.DataFrame, y: np.ndarray, genes: list[str], labels):
    order = np.argsort(y, kind="stable")
    Z = (X[genes] - X[genes].mean()) / X[genes].std()
    Z = Z.iloc[order].clip(-3, 3)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.heatmap(Z, cmap="vlag", center=0, ax=ax, yticklabels=False, cbar_kws={"label": "z-score"})
    split = int((y == 0).sum())
    ax.axhline(split, color=INK, lw=2)
    ax.set_ylabel(f"Samples  ({labels[0]} above line, {labels[1]} below)")
    ax.set_title("Expression of top genes across samples")
    fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(min(0.32 * n + 2, 11), min(0.3 * n + 1.6, 10)))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Correlation (r)", "shrink": 0.7})
    ax.set_title("Gene–gene correlation")
    fig.tight_layout()
    return fig


def network(G: nx.Graph, hubs: pd.DataFrame, top_hubs: int = 5):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    H = G.subgraph([n for n in G.nodes if G.degree(n) > 0])
    if H.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No gene pairs pass this threshold.\nLower the threshold to see connections.",
                ha="center", va="center", color=INK)
        ax.axis("off")
        return fig
    # lay out each connected cluster on its own, then arrange clusters in a grid
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    cols = int(np.ceil(np.sqrt(len(comps))))
    pos = {}
    for i, comp in enumerate(comps):
        sub = H.subgraph(comp)
        p = nx.circular_layout(sub) if len(comp) <= 3 else nx.kamada_kawai_layout(sub)
        scale = 0.35 + 0.12 * np.sqrt(len(comp))
        ox, oy = (i % cols) * 2.6, -(i // cols) * 2.6
        for n, (x, y) in p.items():
            pos[n] = (ox + scale * x, oy + scale * y)
    top = set(hubs.head(top_hubs)["gene"])
    deg = dict(H.degree())
    ecol = [ROSE if d["weight"] > 0 else TEAL for _, _, d in H.edges(data=True)]
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color=ecol, alpha=0.45,
                           width=[1 + 2 * abs(d["weight"]) for _, _, d in H.edges(data=True)])
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[180 + 90 * deg[n] for n in H.nodes],
                           node_color=[ROSE if n in top else PALE for n in H.nodes],
                           edgecolors=INK, linewidths=0.8)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=8,
                            font_color=INK, font_weight="bold")
    ax.set_title("Gene interaction network (hubs filled)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def hub_bar(hubs: pd.DataFrame, top: int = 10):
    d = hubs.head(top)
    fig, ax = _fig(6.2, 3.6)
    ax.bar(d["gene"], d["connections"], color=ROSE)
    ax.set_ylabel("Connections (degree)")
    ax.set_title("Top hub genes")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def enrichment_bar(res: pd.DataFrame, top: int = 10):
    d = res.head(top).iloc[::-1]
    fig, ax = _fig(7.0, 0.38 * len(d) + 1.3)
    ax.barh(d["Term"], d["-log10(adj p)"], color=ROSE)
    ax.axvline(-np.log10(0.05), color=MUTE, ls="--", lw=1)
    ax.set_xlabel("-log10(adjusted p)   (dashed line = p 0.05)")
    ax.set_title("KEGG pathway enrichment")
    fig.tight_layout()
    return fig


def km_plot(res: dict, gene: str, time_unit: str = "days"):
    fig, ax = _fig(6.0, 4.0)
    for key, c, lab in (("high", ROSE, "High"), ("low", TEAL, "Low")):
        t, s = res[key]
        ax.step(t, s, where="post", color=c, lw=2, label=f"{lab} {gene} (n = {res['n_' + key]})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Survival probability")
    p = res["p"]
    ax.set_title(f"Overall survival by {gene} expression (log-rank p = {p:.2g})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ HRV
def tachogram(rr, title="RR interval series", rr2=None, labels=("Recording", "")):
    fig, ax = _fig(7.0, 3.0)
    t = np.cumsum(rr) / 1000
    ax.plot(t, rr, color=ROSE, lw=1.2, label=labels[0])
    if rr2 is not None:
        ax.plot(np.cumsum(rr2) / 1000, rr2, color=TEAL, lw=1.2, label=labels[1])
        ax.legend(frameon=False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RR interval (ms)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def rr_histogram(rr, rr2=None, labels=("Recording", "")):
    fig, ax = _fig(4.6, 3.4)
    ax.hist(rr, bins=30, color=ROSE, alpha=0.75, label=labels[0])
    if rr2 is not None:
        ax.hist(rr2, bins=30, color=TEAL, alpha=0.6, label=labels[1])
        ax.legend(frameon=False)
    ax.set_xlabel("RR interval (ms)")
    ax.set_ylabel("Beats")
    ax.set_title("RR distribution")
    fig.tight_layout()
    return fig


def psd_plot(freq_res, freq_res2=None, labels=("Recording", "")):
    fig, ax = _fig(4.6, 3.4)
    for fr, c, lab in ((freq_res, ROSE, labels[0]), (freq_res2, TEAL, labels[1])):
        if fr is None:
            continue
        f, p = fr["_freq"], fr["_psd"]
        m = f <= 0.5
        ax.plot(f[m], p[m], color=c, lw=1.6, label=lab if freq_res2 is not None else None)
    ax.axvspan(0.04, 0.15, color=SAND, alpha=0.15, label="LF band")
    ax.axvspan(0.15, 0.40, color=TEAL, alpha=0.10, label="HF band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (ms²/Hz)")
    ax.set_title("Power spectrum")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def poincare_plot(rr, pc, rr2=None, labels=("Recording", "")):
    fig, ax = _fig(4.4, 4.2)
    ax.scatter(rr[:-1], rr[1:], s=8, color=ROSE, alpha=0.55, label=labels[0])
    if rr2 is not None:
        ax.scatter(rr2[:-1], rr2[1:], s=8, color=TEAL, alpha=0.55, label=labels[1])
        ax.legend(frameon=False)
    lo = min(rr.min(), rr2.min() if rr2 is not None else rr.min()) - 20
    hi = max(rr.max(), rr2.max() if rr2 is not None else rr.max()) + 20
    ax.plot([lo, hi], [lo, hi], color=MUTE, lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("RRₙ (ms)")
    ax.set_ylabel("RRₙ₊₁ (ms)")
    ax.set_title(f"Poincaré plot  (SD1 {pc['SD1 (ms)']:.1f}, SD2 {pc['SD2 (ms)']:.1f})")
    fig.tight_layout()
    return fig


def stress_gauge(score: float):
    fig, ax = plt.subplots(figsize=(6.4, 1.3))
    bands = [(0, 40, "#CFE5E8", "Relaxed"), (40, 60, "#EFE7EC", "Typical"),
             (60, 75, "#F2D3C4", "Elevated"), (75, 100, "#E7B3C7", "High")]
    for lo, hi, c, lab in bands:
        ax.barh(0, hi - lo, left=lo, color=c, height=0.6)
        ax.text((lo + hi) / 2, -0.55, lab, ha="center", va="top", fontsize=8.5, color=INK)
    if np.isfinite(score):
        ax.plot([score, score], [-0.35, 0.35], color=INK, lw=3)
        ax.text(score, 0.45, f"{score:.0f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    fig.tight_layout()
    return fig


def contribution_bar(contrib: pd.DataFrame):
    d = contrib.copy()
    d["points"] = d["points"].astype(float)
    d = d.sort_values("points")
    fig, ax = _fig(5.8, 0.5 * len(d) + 1.2)
    ax.barh(d.index, d["points"], color=[ROSE if v > 0 else TEAL for v in d["points"]])
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Points added to (+) or removed from (−) the score")
    ax.set_title("Why this score?")
    fig.tight_layout()
    return fig


def thesis_cohort(base: pd.DataFrame, you: dict | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, col in zip(axes, ["SDNN (ms)", "RMSSD (ms)"]):
        ax.scatter(np.random.default_rng(0).normal(0, 0.05, len(base)), base[col],
                   color=MUTE, s=40, label="Thesis participants (n = 10)")
        ax.axhline(base[col].mean(), color=MUTE, ls="--", lw=1)
        if you and you.get(col) is not None:
            ax.scatter([0], [you[col]], color=ROSE, s=110, marker="D", label="This recording", zorder=3)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(col)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig
