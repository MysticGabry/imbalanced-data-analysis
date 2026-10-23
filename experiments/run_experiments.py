"""
Per ogni dataset, campionatore e classificatore esegue una 5-fold
stratificata ripetuta 2 volte e salva in ``results/``:

- ``risultati_per_fold.csv``  : una riga per fold;
- ``risultati.csv``           : media e deviazione standard;
- ``riepilogo.md``            : tabelle pronte da incollare nella tesi;
- ``heatmap_<metrica>_<clf>.png``.

Uso:  python experiments/run_experiments.py
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import SEED, describe, evaluate, load_datasets, make_samplers  # noqa: E402
from plots import heatmap  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "gmean", "roc_auc"]

CLASSIFIERS = {
    "LogReg": LogisticRegression(max_iter=2000),
    "AlberoDecisione": DecisionTreeClassifier(random_state=SEED),
    "kNN": KNeighborsClassifier(n_neighbors=5),
}


def main():
    RESULTS.mkdir(exist_ok=True)
    datasets = load_datasets()
    rows = []
    for ds_name, (X, y, cat_cols) in datasets.items():
        print(f"\n== {ds_name}  [{X.shape[0]}x{X.shape[1]}]  classi: {describe(y)}")
        samplers = make_samplers()
        if cat_cols:
            samplers["SMOTENC"] = SMOTENC(categorical_features=cat_cols, random_state=SEED)
        for s_name, sampler in samplers.items():
            for c_name, clf in CLASSIFIERS.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        fold_rows = evaluate(
                            X, y, cat_cols, sampler, clf, smotenc=(s_name == "SMOTENC")
                        )
                    except (ValueError, RuntimeError) as exc:
                        print(f"   {s_name:<17} {c_name:<16} SALTATO: {exc}")
                        continue
                for r in fold_rows:
                    r.update(dataset=ds_name, sampler=s_name, classifier=c_name)
                rows.extend(fold_rows)
                gm = pd.DataFrame(fold_rows)["gmean"].mean()
                print(f"   {s_name:<17} {c_name:<16} G-mean {gm:.3f}")

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(RESULTS / "risultati_per_fold.csv", index=False)

    keys = ["dataset", "sampler", "classifier"]
    agg = per_fold.groupby(keys, sort=False)[
        METRICS + ["invalid_categorical_values", "resample_seconds", "n_train_resampled"]
    ].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(RESULTS / "risultati.csv", index=False)

    write_summary(agg, datasets)
    sampler_order = list(dict.fromkeys(agg["sampler"]))
    for metric in ["gmean", "balanced_accuracy"]:
        for c_name in CLASSIFIERS:
            sub = agg[agg["classifier"] == c_name]
            table = sub.pivot(index="sampler", columns="dataset", values=f"{metric}_mean")
            table = table.reindex(index=sampler_order, columns=list(datasets))
            heatmap(
                table,
                title=f"{metric} medio — {c_name}",
                path=RESULTS / f"heatmap_{metric}_{c_name}.png",
            )
    print(f"\nRisultati salvati in {RESULTS}")


def write_summary(agg, datasets):
    lines = ["# Riepilogo risultati", ""]
    lines.append("Protocollo: 5-fold stratificata ripetuta 2 volte; ricampionamento "
                 "applicato solo al training set di ogni fold. Valori: media ± dev. std.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("| Dataset | Campioni | Feature | Distribuzione classi |")
    lines.append("|---|---|---|---|")
    for name, (X, y, _) in datasets.items():
        lines.append(f"| {name} | {X.shape[0]} | {X.shape[1]} | {describe(y)} |")
    lines.append("")

    for metric in ["gmean", "balanced_accuracy", "f1_macro", "roc_auc", "accuracy"]:
        lines.append(f"## {metric}")
        lines.append("")
        for c_name in CLASSIFIERS:
            sub = agg[agg["classifier"] == c_name]
            lines.append(f"### {c_name}")
            lines.append("")
            ds = list(datasets)
            lines.append("| Campionatore | " + " | ".join(ds) + " |")
            lines.append("|---|" + "---|" * len(ds))
            best = {d: sub[sub["dataset"] == d][f"{metric}_mean"].max() for d in ds}
            for s in dict.fromkeys(sub["sampler"]):
                cells = []
                for d in ds:
                    r = sub[(sub["sampler"] == s) & (sub["dataset"] == d)]
                    if r.empty:
                        cells.append("—")
                        continue
                    m, sd = r[f"{metric}_mean"].iloc[0], r[f"{metric}_std"].iloc[0]
                    cell = f"{m:.3f} ± {sd:.3f}"
                    cells.append(f"**{cell}**" if abs(m - best[d]) < 1e-12 else cell)
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")

    for name, (_, _, cat_cols) in datasets.items():
        if not cat_cols:
            continue
        mixed = agg[(agg["dataset"] == name) & (agg["classifier"] == "LogReg")]
        lines.append(f"## Valori categorici non validi generati ({name})")
        lines.append("")
        lines.append("Numero medio, per fold, di celle one-hot dei campioni sintetici "
                     "diverse da 0 e 1 (cioè categorie \"miscelate\" senza significato).")
        lines.append("")
        lines.append("| Campionatore | Valori non validi | Campioni di training |")
        lines.append("|---|---|---|")
        for _, r in mixed.iterrows():
            lines.append(f"| {r['sampler']} | {r['invalid_categorical_values_mean']:.0f} "
                         f"| {r['n_train_resampled_mean']:.0f} |")
        lines.append("")
    (RESULTS / "riepilogo.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
