"""Figures for the technical report, generated from experiments/results/*.json.

    python make_figures.py        # writes paper/figures/*.pdf (and .png previews)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGS = HERE.parents[1] / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

BUDGETS = ["0", "1", "3", "7", "15", "31"]
# categorical palette (fixed slot order) + marker shapes as secondary encoding for print
STYLE = {
    "const": dict(color="#52514e", marker="x", label="constant 0.5", ls="-"),
    "empirical_mean": dict(color="#eb6834", marker="s", label="empirical mean (organisers)", ls="-"),
    "prior": dict(color="#eda100", marker="^", label="prior only", ls="-"),
    "shrink": dict(color="#1baf7a", marker="D", label="prior + shrinkage", ls="--"),
    "full": dict(color="#2a78d6", marker="o", label="full (shipped)", ls="-"),
    "oracle_rate": dict(color="#6250d6", marker="", label="oracle: pair rate", ls="--"),
    "oracle_p": dict(color="#e87ba4", marker="", label="oracle: true probability", ls=":"),
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": TEXT,
                     "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def brier_curves(ax, summary, title, methods):
    x = [math.log2(int(b) + 1) for b in BUDGETS]
    for m in methods:
        if m not in summary["methods"]:
            continue
        y = [summary["methods"][m]["brier_by_budget"][b] for b in BUDGETS]
        st = STYLE[m]
        ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.6, ms=5,
                markeredgecolor="white", markeredgewidth=0.8, label=st["label"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(BUDGETS)
    ax.set_xlabel("acquired labels $n$ (axis: $\\log_2(n+1)$)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title(title, loc="left", fontsize=9, color=TEXT)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_brier_vs_budget():
    syn, real = load("synthetic_main.json"), load("real_testbed_main.json")
    if syn is None and real is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    methods = ["const", "empirical_mean", "prior", "shrink", "full", "oracle_rate", "oracle_p"]
    if syn:
        brier_curves(axes[0], syn, f"(a) synthetic IRT testbed, {syn['n_pairs']} pairs", methods)
    if real:
        brier_curves(axes[1], real, f"(b) real Ollama testbed, {real['n_pairs']} pairs", methods)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "brier_vs_budget.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "brier_vs_budget.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_signal_sweep():
    sweep = load("synthetic_signal_sweep.json")
    if sweep is None:
        return
    settings = [s for s in sweep["settings"] if s["gamma_length"] == 0.3]
    settings.sort(key=lambda s: s["sigma_topic"])
    x = [s["sigma_topic"] for s in settings]
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.7), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    for m in ("shrink", "full", "oracle_rate", "oracle_p"):
        if m not in settings[0]["methods"]:
            continue
        y = [s["methods"][m]["alc"]["mean"] for s in settings]
        st = STYLE[m]
        ax.plot(x, y, color=st["color"], marker=st["marker"] or None, ls=st["ls"], lw=1.6, ms=4,
                markeredgecolor="white", markeredgewidth=0.8, label=st["label"], zorder=3)
    ax.set_xlabel("$\\sigma_{\\mathrm{topic}}$ (logits)")
    ax.set_ylabel("Brier ALC")
    ax.set_title("(a) ALC and oracles", loc="left", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    # paired difference full - shrinkage with 95% CI (negative favours the full model)
    ax = axes[1]
    d = [s["paired"]["full_vs_shrink"] for s in settings]
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.fill_between(x, [v["ci_lo"] * 1000 for v in d], [v["ci_hi"] * 1000 for v in d], color=STYLE["full"]["color"], alpha=0.18, lw=0)
    ax.plot(x, [v["mean"] * 1000 for v in d], color=STYLE["full"]["color"], marker="o", lw=1.6, ms=4,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.set_xlabel("$\\sigma_{\\mathrm{topic}}$ (logits)")
    ax.set_ylabel("full $-$ shrinkage ($\\times 10^{-3}$ ALC)")
    ax.set_title("(b) paired difference, 95% CI", loc="left", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.02), handlelength=1.8)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.13, 1, 1))
    fig.savefig(FIGS / "signal_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "signal_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_alpha0():
    syn, real = load("alpha0_sensitivity_synthetic.json"), load("alpha0_sensitivity_real.json")
    if syn is None and real is None:
        return
    fig, ax = plt.subplots(figsize=(2.9, 2.7))
    for data, key, color, marker, label in ((syn, "synthetic", "#2a78d6", "o", "synthetic"),
                                            (real, "real", "#eb6834", "s", "real Ollama")):
        if data is None:
            continue
        alphas = sorted(data[key], key=float)
        for method, ls in (("shrink", "--"), ("full", "-")):
            y = [data[key][a][method]["alc"]["mean"] for a in alphas]
            ax.plot([float(a) for a in alphas], y, color=color, marker=marker, ls=ls, lw=1.6, ms=5,
                    markeredgecolor="white", markeredgewidth=0.8, label=f"{label}, {method}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("prior strength $\\alpha_0$ (pseudo-counts)")
    ax.set_ylabel("Brier ALC")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "alpha0_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "alpha0_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


XP_STYLE = {
    "XP": dict(color="#2a78d6", marker="o", label="XP: cross-pair model (shipped)", ls="-"),
    "S0": dict(color="#eb6834", marker="s", label="S0: rounds 1-3 predictor on the global list", ls="-"),
    "S1": dict(color="#1baf7a", marker="D", label="S1: same, own pair's labels only", ls="--"),
    "oracle_rate": dict(color="#6250d6", marker="", label="oracle: pair success rate", ls="--"),
}


def fig_xp_budget():
    cv, add = load("round4/xpair_cv.json"), load("round4/xpair_addendum1.json")
    if cv is None or add is None:
        return
    panels = [(cv["configs"]["A"], "(a) config A: formative-like, %d pairs" % cv["configs"]["A"]["n_units"]),
              (add["configs"]["E"], "(b) config E: distinct models, %d pairs" % add["configs"]["E"]["n_units"])]
    x = [math.log2(int(b) + 1) for b in BUDGETS]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (summ, title) in zip(axes, panels):
        for m in ("S0", "S1", "XP", "oracle_rate"):
            rec = summ["methods"].get(m)
            if rec is None:
                continue
            st = XP_STYLE[m]
            y = rec["brier"]
            if m in ("XP", "S0"):
                lo = [c["ci_lo"] for c in rec["brier_ci"]]
                hi = [c["ci_hi"] for c in rec["brier_ci"]]
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.14, lw=0, zorder=1)
            ax.plot(x, y, color=st["color"], marker=st["marker"] or None, ls=st["ls"], lw=1.6, ms=5,
                    markeredgecolor="white", markeredgewidth=0.8, label=st["label"], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGETS)
        ax.set_xlabel(r"labels per pair $n$ (axis: $\log_2(n+1)$)")
        ax.set_title(title, loc="left", fontsize=8.5, color=TEXT)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Brier score (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGS / "xp_budget.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "xp_budget.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_brier_vs_budget()
    fig_signal_sweep()
    fig_alpha0()
    fig_xp_budget()
    print("figures written to", FIGS)
