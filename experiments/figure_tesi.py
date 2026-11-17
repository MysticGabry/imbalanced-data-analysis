# Rigenera le figure del Capitolo 3 a partire da risultati.csv e stampa i ranghi medi.
# Le etichette dei metodi sono quelle del CSV (il trattamento di controllo è
# "Valore iniziale"); i numeri usano la virgola decimale.
# Uso: python -I genera_figure.py <cartella results> <cartella figure>
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

res, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(os.path.join(res, "risultati.csv"))

BASE = "Valore iniziale"
BLUE = "#2a78d6"
GREY = "#b8b8ae"
LIGHT = "#e4e4de"
TEXT = "#4a4a48"
DATASETS = ["satimage", "mammography", "abalone 19", "page blocks", "bank marketing"]
DS_NAMES = ["Satimage", "Mammography", "Abalone 19", "Page Blocks", "Bank Marketing"]
CLFS = [("LogReg", "Regressione logistica"), ("AlberoDecisione", "Albero di decisione"),
        ("kNN", "k-NN")]
ORDER = list(dict.fromkeys(df["sampler"]))          # ordine del CSV
COMMON = [s for s in ORDER if s != "SMOTENC"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": GREY, "xtick.color": TEXT, "ytick.color": TEXT,
                     "axes.labelcolor": TEXT})


def it(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


comma1 = FuncFormatter(lambda x, _: it(x, 1))


def save(fig, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("scritto", path)


# ---------------------------------------------------------------- heatmap
def heatmap():
    cmap = LinearSegmentedColormap.from_list("blu", ["#cfe1f9", "#3a86e0", "#1e5aa8", "#123a73"])
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 5.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, (clf, title) in zip(axes, CLFS):
        piv = (df[df.classifier == clf].pivot(index="sampler", columns="dataset",
                                              values="gmean_mean").loc[ORDER, DATASETS])
        m = piv.values
        im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        best = np.nanmax(m, axis=0)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                if np.isnan(v):
                    ax.text(j, i, "–", ha="center", va="center", color=TEXT, fontsize=8)
                    continue
                bold = abs(v - best[j]) < 5e-5
                ax.text(j, i, it(v, 2), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.5 else "black",
                        fontweight="bold" if bold else "normal")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DS_NAMES, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks(np.arange(-.5, len(DATASETS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(ORDER)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.75)
    cb.set_label("G-mean media")
    cb.ax.yaxis.set_major_formatter(comma1)
    cb.outline.set_visible(False)
    save(fig, "fig_heatmap_gmean.png")


# ------------------------------------------------ accuratezza contro G-mean
def acc_gmean():
    d = df[(df.dataset == "satimage") & (df.classifier == "LogReg") & df.sampler.isin(COMMON)]
    d = d.sort_values("gmean_mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(d))
    for yi, a, g in zip(y, d.accuracy_mean, d.gmean_mean):
        ax.plot([a, g], [yi, yi], color=LIGHT, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter(d.accuracy_mean, y, s=42, color=GREY, zorder=2, label="Accuratezza",
               edgecolor="white", linewidth=1)
    ax.scatter(d.gmean_mean, y, s=42, color=BLUE, zorder=3, label="G-mean",
               edgecolor="white", linewidth=1)
    b = d[d.sampler == BASE].iloc[0]
    yb = list(d.sampler).index(BASE)
    ax.text(b.gmean_mean - 0.025, yb, it(b.gmean_mean), ha="right", va="center",
            fontweight="bold", color="black")
    ax.text(b.accuracy_mean + 0.028, yb, it(b.accuracy_mean), ha="left", va="center", color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(d.sampler)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.5, len(d) - 0.5)
    ax.xaxis.set_major_formatter(comma1)
    ax.set_xlabel("Valore medio sui 10 fold")
    ax.grid(axis="x", color=LIGHT, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, "fig_accuratezza_gmean.png")


# ------------------------------------------------------------ G-mean media
def gmean_media():
    m = df[df.sampler.isin(COMMON)].groupby("sampler").gmean_mean.mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    y = np.arange(len(m))
    colors = [BLUE if s == m.index[-1] else GREY for s in m.index]
    ax.barh(y, m.values, color=colors, height=0.62)
    for yi, (s, v) in zip(y, m.items()):
        top = s == m.index[-1]
        ax.text(v + 0.01, yi, it(v), va="center", ha="left",
                fontweight="bold" if top else "normal", color="black" if top else TEXT,
                fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(m.index)
    ax.set_xlim(0, 0.9)
    ax.xaxis.set_major_formatter(comma1)
    ax.set_xlabel("G-mean media sui 15 casi (5 dataset × 3 classificatori)")
    ax.tick_params(axis="y", length=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    save(fig, "fig_gmean_media.png")


# ------------------------------------------------ rango medio (Tabella delle medie)
def ranghi_medi():
    piv = (df[df.sampler.isin(COMMON)]
           .pivot_table(index=["dataset", "classifier"], columns="sampler",
                        values="gmean_mean")[COMMON])
    mr = piv.rank(axis=1, ascending=False).mean().sort_values()
    print("Rango medio rispetto alla G-mean sui 15 casi:")
    for nome, r in mr.items():
        print(f"  {nome:18s} {it(r, 2)}")


heatmap()
acc_gmean()
gmean_media()
ranghi_medi()
