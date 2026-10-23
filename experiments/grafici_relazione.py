"""Grafici calcolati da results/risultati.csv.

Uso:  python experiments/grafici_relazione.py
"""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plots import GRID, INK, INK_MUTED, plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
ACCENT = "#2a78d6"
MUTED = "#b9b8ae"

DATASET_ESEMPIO = "satimage"


def fmt(v):
    return f"{v:.3f}".replace(".", ",")


def accuratezza_vs_gmean(df):
    sub = df[(df.dataset == DATASET_ESEMPIO) & (df.classifier == "LogReg")]
    sub = sub.sort_values("gmean_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.plot([r.accuracy_mean, r.gmean_mean], [i, i], color=MUTED, lw=2, zorder=1)
    ax.scatter(sub.accuracy_mean, y, s=60, color=MUTED, zorder=2, label="Accuratezza",
               edgecolors="white", linewidths=1.5)
    ax.scatter(sub.gmean_mean, y, s=60, color=ACCENT, zorder=3, label="G-mean",
               edgecolors="white", linewidths=1.5)
    ax.set_yticks(list(y), sub.sampler)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "nessuno":
            lab.set_fontweight("bold")
    base = sub[sub.sampler == "nessuno"].iloc[0]
    i_base = list(sub.sampler).index("nessuno")
    ax.annotate(fmt(base.gmean_mean), (base.gmean_mean, i_base), xytext=(-10, 0),
                textcoords="offset points", ha="right", va="center", fontweight="bold", color=INK)
    ax.annotate(fmt(base.accuracy_mean), (base.accuracy_mean, i_base), xytext=(10, 0),
                textcoords="offset points", ha="left", va="center", color=INK_MUTED)
    best = sub.iloc[-1]
    ax.annotate(fmt(best.gmean_mean), (best.gmean_mean, len(sub) - 1), xytext=(10, 0),
                textcoords="offset points", ha="left", va="center", fontweight="bold", color=INK)
    ax.set_xlim(0, 1.05)
    ax.xaxis.tick_top()
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.set_title("Senza ribilanciare l'accuratezza è alta ma la G-mean è solo "
                 f"{fmt(base.gmean_mean)}\nRegressione logistica, dataset {DATASET_ESEMPIO}, "
                 "media di 10 fold", loc="left", fontsize=10, pad=24)
    fig.tight_layout()
    fig.savefig(RESULTS / "relazione_accuratezza_vs_gmean.png", dpi=200)
    plt.close(fig)


def gmean_media(df):
    sub = df[df.sampler != "SMOTENC"].groupby("sampler").gmean_mean.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [ACCENT if s == sub.index[-1] else MUTED for s in sub.index]
    ax.barh(sub.index, sub.values, color=colors, height=0.6)
    for i, (s, v) in enumerate(sub.items()):
        ax.text(v + 0.008, i, fmt(v), va="center", fontsize=8.5,
                color=INK if s == sub.index[-1] else INK_MUTED,
                fontweight="bold" if s == sub.index[-1] else "normal")
    for lab in ax.get_yticklabels():
        if lab.get_text() in ("nessuno", sub.index[-1]):
            lab.set_fontweight("bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    spread = math.ceil((sub.iloc[-1] - sub.iloc[-7]) * 100) / 100
    ax.set_title(f"{sub.index[-1]} è il migliore in media ({fmt(sub.iloc[-1])}), ma i primi "
                 f"sette metodi distano meno di {spread:.2f}".replace("0.", "0,")
                 + "\nG-mean media su 5 dataset × 3 classificatori",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "relazione_gmean_media.png", dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(RESULTS / "risultati.csv")
    accuratezza_vs_gmean(df)
    gmean_media(df)
    print("Salvati i grafici della relazione in", RESULTS)


if __name__ == "__main__":
    main()
