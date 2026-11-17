"""Verifica di due ipotesi formulate nel Capitolo 3 della tesi.

1. Borderline-SMOTE genera meno valori categorici non validi di SMOTE su
   Abalone 19: dipende dai suoi semi (gli esempi "in pericolo")?
2. Su Mammography la ROC-AUC dell'albero differisce dalla balanced accuracy:
   dipende da esempi con feature identiche ma classi diverse?

Usa lo stesso protocollo dell'esperimento (stessi fold, stesso seme).

Uso:  python experiments/verifiche_ipotesi.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import SEED, _binary_benchmark, _preprocessor, load_abalone_19, load_bank_marketing  # noqa: E402

warnings.simplefilter("ignore")
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)


def danger_mask(X, y, m_neighbors=10):
    """Esempi minoritari 'in pericolo' secondo Borderline-SMOTE (borderline-1):
    almeno metà, ma non tutti, degli m vicini sono maggioritari."""
    nn = NearestNeighbors(n_neighbors=m_neighbors + 1).fit(X)
    Xmin = X[y == 1]
    idx = nn.kneighbors(Xmin, return_distance=False)[:, 1:]
    n_maj = (y[idx] != 1).sum(axis=1)
    return (n_maj >= m_neighbors / 2) & (n_maj < m_neighbors), n_maj


def borderline(name, loader):
    X, y, cat_cols = loader()
    n_num = X.shape[1] - len(cat_cols)
    rows = []
    for tr, _ in CV.split(X, y):
        pre = _preprocessor(cat_cols, X.shape[1]).fit(X[tr])
        Xt, yt = pre.transform(X[tr]), y[tr]
        danger, n_maj = danger_mask(Xt, yt)
        Xmin = Xt[yt == 1]
        ohe = Xmin[:, n_num:]

        def stessa_categoria(semi):
            """Quota di coppie seme-vicino minoritario (k=5) con tutte le
            categoriche uguali: in quel caso l'interpolazione è valida."""
            if semi.sum() == 0:
                return np.nan
            nn = NearestNeighbors(n_neighbors=6).fit(Xmin)
            idx = nn.kneighbors(Xmin[semi], return_distance=False)[:, 1:]
            eq = [(np.isclose(ohe[idx[i]], ohe[semi][i]).all(axis=1)).mean()
                  for i in range(idx.shape[0])]
            return float(np.mean(eq))

        out = {"minoritari": int((yt == 1).sum()), "in_pericolo": int(danger.sum()),
               "rumore (tutti i vicini maggioritari)": int((n_maj == 10).sum()),
               "coppie stessa categoria, tutti i semi": stessa_categoria(np.ones(len(Xmin), bool)),
               "coppie stessa categoria, semi in pericolo": stessa_categoria(danger)}
        for lab, s in [("SMOTE", SMOTE(random_state=SEED)),
                       ("Borderline", BorderlineSMOTE(random_state=SEED))]:
            Xr, _ = s.fit_resample(Xt, yt)
            syn = Xr[len(Xt):, n_num:]
            bad = ~np.isclose(syn, 0) & ~np.isclose(syn, 1)
            out[f"sintetici {lab}"] = len(syn)
            out[f"celle non valide {lab}"] = int(bad.sum())
            out[f"sintetici non validi {lab} (%)"] = 100 * bad.any(axis=1).mean() if len(syn) else np.nan
        rows.append(out)
    d = pd.DataFrame(rows)
    print(f"\n=== Borderline-SMOTE su {name}: valori per fold ===")
    print(d.round(3).to_string())
    print("\nMedia sui 10 fold:")
    print(d.mean().round(3).to_string())


def mammography():
    X, y, _ = _binary_benchmark("mammography")
    df = pd.DataFrame(X)
    df["y"] = y
    g = df.groupby(list(range(X.shape[1])))["y"].agg(["size", "nunique", "sum"])
    dup = g[g["size"] > 1]
    conf = g[g["nunique"] > 1]
    print("\n=== Mammography: duplicati nell'intero dataset ===")
    print("esempi:", len(df), "| righe di feature distinte:", len(g))
    print("gruppi di duplicati:", len(dup), "| esempi coinvolti:", int(dup["size"].sum()))
    print("gruppi con classi diverse:", len(conf), "| esempi coinvolti:", int(conf["size"].sum()),
          "| di cui positivi:", int(conf["sum"].sum()))

    rows = []
    for tr, te in CV.split(X, y):
        pre = _preprocessor([], X.shape[1]).fit(X[tr])
        Xt, Xe = pre.transform(X[tr]), pre.transform(X[te])
        tree = DecisionTreeClassifier(random_state=SEED).fit(Xt, y[tr])
        leaves = tree.apply(Xt)
        val = tree.tree_.value[:, 0, :]
        frac = val / val.sum(axis=1, keepdims=True)
        impure_nodes = {n for n in np.unique(leaves) if 0 < frac[n, 1] < 1}
        p = tree.predict_proba(Xe)[:, 1]
        d = pd.DataFrame(Xt)
        d["y"] = y[tr]
        gg = d.groupby(list(range(Xt.shape[1])))["y"].nunique()
        rows.append({"foglie": int(tree.get_n_leaves()), "foglie impure": len(impure_nodes),
                     "gruppi duplicati con classi diverse (train)": int((gg > 1).sum()),
                     "esempi di test con probabilità frazionaria": int(((p > 0) & (p < 1)).sum())})
    d = pd.DataFrame(rows)
    print("\nAlbero di decisione sui dati originali, per fold:")
    print(d.to_string())
    print("\nMedia:")
    print(d.mean().round(2).to_string())


if __name__ == "__main__":
    borderline("Abalone 19", load_abalone_19)
    borderline("Bank Marketing", load_bank_marketing)
    mammography()
