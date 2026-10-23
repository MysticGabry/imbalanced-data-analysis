import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#1f1f1e"
INK_MUTED = "#6b6a63"
GRID = "#e4e3dc"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# ordine palette: maggioritaria, minoritaria, terza classe.
CLASS_COLORS = ["#00ff04", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.titlecolor": INK,
})


def heatmap(table, title, path):
    """Tabella campionatore x dataset colorata per valore, con i numeri in cella."""
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.5 * values.shape[1] + 2.4, 0.42 * values.shape[0] + 1.4))
    finite = values[np.isfinite(values)]
    im = ax.imshow(values, cmap=BLUES, vmin=finite.min(), vmax=finite.max(), aspect="auto")
    ax.set_xticks(range(values.shape[1]), table.columns, rotation=20, ha="right")
    ax.set_yticks(range(values.shape[0]), table.index)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)

    mid = (finite.min() + finite.max()) / 2
    for i in range(values.shape[0]):
        best = np.nanmax(values[:, :], axis=0)
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", color=INK_MUTED)
                continue
            weight = "bold" if np.isclose(v, best[j]) else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > mid else INK, fontweight=weight)
    ax.set_title(title + "  (grassetto = migliore per dataset)", loc="left", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def scatter_grid(panels, path, ncols=4):
    """Griglia di scatter 2D: ``panels`` è una lista di (titolo, X, y, n_orig).

    I campioni con indice >= n_orig sono sintetici e vengono disegnati con
    un marker diverso (crocetta) oltre che con il colore della classe.
    """
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, X, y, n_orig) in zip(axes, panels):
        classes = np.unique(y)
        for k, cls in enumerate(classes):
            color = CLASS_COLORS[k % len(CLASS_COLORS)]
            real = (y == cls) & (np.arange(y.size) < n_orig)
            synth = (y == cls) & (np.arange(y.size) >= n_orig)
            ax.scatter(X[real, 0], X[real, 1], s=9, color=color, alpha=0.55,
                       linewidths=0.4, edgecolors="white", label=f"classe {cls}")
            if synth.any():
                ax.scatter(X[synth, 0], X[synth, 1], s=12, color=color, marker="x",
                           linewidths=0.8, label=f"classe {cls} (sintetici)")
        counts = ", ".join(f"{c}:{(y == c).sum()}" for c in classes)
        ax.set_title(f"{title}\n{counts}", loc="left", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(panels):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:len(panels)]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
