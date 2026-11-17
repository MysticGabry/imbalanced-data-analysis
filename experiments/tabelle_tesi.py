"""Genera le tabelle LaTeX della tesi a partire da results/risultati.csv.

Tabelle prodotte (in results/tabelle/):
- tab_gmean_<clf>.tex : G-mean media ± dev. std. per dataset;
- tab_medie.tex   : media di ogni metrica sui 15 casi e rango medio;
- tab_tempi.tex   : dimensione del training set e tempo di ricampionamento;
- tab_smotenc.tex : valori categorici non validi e G-mean sui dataset misti.

Uso:  python experiments/tabelle_tesi.py
"""

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "tabelle"

DATASETS = ["satimage", "mammography", "abalone 19", "page blocks", "bank marketing"]
INTESTAZIONE = (r"\textbf{Metodo} & \textbf{Satimage} & \textbf{Mammogr.} & "
                r"\textbf{Abalone 19} & \textbf{Page Bl.} & \textbf{Bank Mkt.} \\")
CLASSIFICATORI = ["LogReg", "AlberoDecisione", "kNN"]
BASE = "Valore iniziale"
# Gruppi di righe separati da una linea orizzontale nelle tabelle per dataset.
GRUPPI = [[BASE], ["ROS", "SMOTE", "Borderline-SMOTE", "ADASYN"], ["RUS", "NearMiss-1"],
          ["Tomek", "ENN"], ["SMOTE+ENN", "SMOTE+Tomek"], ["SMOTENC"]]


def dec(x, nd=3):
    """Numero con la virgola decimale nel formato LaTeX della tesi: 0{,}123."""
    return f"{x:.{nd}f}".replace(".", "{,}")


def intero(x):
    """Intero con lo spazio sottile come separatore delle migliaia: 5\\,148."""
    return f"{round(x):,}".replace(",", r"\,")


def grassetto(testo, attivo):
    return rf"\textbf{{{testo}}}" if attivo else testo


def tabella(colonne, intestazione, righe):
    return "\n".join([rf"\begin{{tabular}}{{@{{}}{colonne}@{{}}}}", r"\toprule", intestazione,
                      r"\midrule", *righe, r"\bottomrule", r"\end{tabular}", ""])


def per_dataset(df, metrica, clf):
    """Media ± dev. std. di una metrica, in grassetto il migliore di ogni dataset."""
    sub = df[df.classifier == clf]
    media = sub.pivot(index="sampler", columns="dataset", values=f"{metrica}_mean")
    std = sub.pivot(index="sampler", columns="dataset", values=f"{metrica}_std")
    # il migliore si decide sui valori come appaiono in tabella (tre decimali)
    migliore = media.round(3).max()
    righe = []
    for g, gruppo in enumerate(GRUPPI):
        if g > 0:
            righe.append(r"\midrule")
        for s in gruppo:
            celle = []
            for d in DATASETS:
                m = media.loc[s, d] if s in media.index else float("nan")
                if pd.isna(m):
                    celle.append("--")
                    continue
                v = grassetto(dec(m), round(m, 3) == migliore[d])
                celle.append(rf"{v} \scriptsize{{$\pm$ {dec(std.loc[s, d])}}}")
            righe.append(f"{s} & " + " & ".join(celle) + r" \\")
    return tabella("lrrrrr", INTESTAZIONE, righe)


def medie(df):
    """Media di ciascuna metrica sui 15 casi e rango medio rispetto alla G-mean."""
    comuni = df[df.sampler != "SMOTENC"]
    metriche = ["accuracy", "balanced_accuracy", "f1_macro", "gmean", "roc_auc"]
    m = comuni.groupby("sampler")[[f"{x}_mean" for x in metriche]].mean()
    # in ogni caso (dataset, classificatore) si ordinano i metodi per G-mean:
    # rango 1 al migliore, rango medio in caso di parità; poi si fa la media
    piv = comuni.pivot_table(index=["dataset", "classifier"], columns="sampler",
                             values="gmean_mean")
    m["rango"] = piv.rank(axis=1, ascending=False).mean()
    m = m.sort_values("gmean_mean", ascending=False)
    arrot = m.round({**{f"{x}_mean": 3 for x in metriche}, "rango": 2})
    migliore = {c: (arrot[c].min() if c == "rango" else arrot[c].max()) for c in m.columns}
    righe = []
    for s, r in m.iterrows():
        celle = [grassetto(dec(r[c], 2 if c == "rango" else 3), arrot.loc[s, c] == migliore[c])
                 for c in m.columns]
        righe.append(f"{s} & " + " & ".join(celle) + r" \\")
    intest = (r"\textbf{Metodo} & \textbf{Acc.} & \textbf{Bal.\ acc.} & \textbf{F1 macro} & "
              r"\textbf{G-mean} & \textbf{ROC-AUC} & \textbf{Rango} \\")
    return tabella("lrrrrrr", intest, righe)


def tempi(df):
    """Dimensione media del training set e tempo medio di ricampionamento per fold.

    Il ricampionamento viene ripetuto per ciascun classificatore: come nella
    tabella dei valori categorici, si usano i valori misurati nelle prove
    con la regressione logistica.
    """
    t = (df[df.classifier == "LogReg"]
         .set_index(["sampler", "dataset"])[["n_train_resampled_mean", "resample_seconds_mean"]])
    ordine = [s for gruppo in GRUPPI for s in gruppo]
    righe = []
    for s in ordine:
        celle = []
        for d in DATASETS:
            if (s, d) not in t.index:
                celle.append("--")
                continue
            n, sec = t.loc[(s, d)]
            if s == BASE:
                tempo = "--"
            elif sec < 0.005:
                tempo = r"$<$0{,}01"
            else:
                tempo = dec(sec, 2)
            celle.append(f"{intero(n)} / {tempo}")
        righe.append(f"{s} & " + " & ".join(celle) + r" \\")
        if s == BASE:
            righe.append(r"\midrule")
    return tabella("lrrrrr", INTESTAZIONE, righe)


def smotenc(df):
    """Celle one-hot non valide (per fold) e G-mean sui dataset con attributi categorici."""
    metodi = ["SMOTE", "Borderline-SMOTE", "ADASYN", "SMOTE+ENN", "SMOTE+Tomek", "SMOTENC"]
    nomi = {"abalone 19": "Abalone 19", "bank marketing": "Bank Marketing"}
    righe = []
    for i, (d, nome) in enumerate(nomi.items()):
        if i > 0:
            righe.append(r"\midrule")
        for j, s in enumerate(metodi):
            sub = df[(df.dataset == d) & (df.sampler == s)].set_index("classifier")
            non_validi = sub.loc["LogReg", "invalid_categorical_values_mean"]
            gm = [dec(sub.loc[c, "gmean_mean"]) for c in CLASSIFICATORI]
            prima = rf"\multirow{{{len(metodi)}}}{{*}}{{{nome}}}" if j == 0 else ""
            righe.append(f"{prima} & {s} & {intero(non_validi)} & " + " & ".join(gm) + r" \\")
    intest = "\n".join([
        r"& & \textbf{Celle one-hot} & \multicolumn{3}{c}{\textbf{G-mean}} \\",
        r"\cmidrule(l){4-6}",
        r"\textbf{Dataset} & \textbf{Metodo} & \textbf{non valide} & \textbf{LogReg} & "
        r"\textbf{Albero} & \textbf{$k$-NN} \\"])
    return tabella("llrrrr", intest, righe)


def main():
    df = pd.read_csv(RESULTS / "risultati.csv")
    OUT.mkdir(exist_ok=True)
    tabelle = {"tab_medie.tex": medie(df), "tab_tempi.tex": tempi(df),
               "tab_smotenc.tex": smotenc(df)}
    for clf in CLASSIFICATORI:
        tabelle[f"tab_gmean_{clf}.tex"] = per_dataset(df, "gmean", clf)
    for nome, testo in tabelle.items():
        (OUT / nome).write_text(testo, encoding="utf-8", newline="\n")
        print("scritta", OUT / nome)


if __name__ == "__main__":
    main()
