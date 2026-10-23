"""Dataset, campionatori e protocollo di valutazione condivisi dagli esperimenti.

Regola fondamentale: il ricampionamento si applica SOLO al training set di
ogni fold. Ricampionare prima della cross-validation fa finire punti
sintetici (o copie) nel test set e gonfia artificialmente i risultati.
"""

import time

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)

SEED = 42


# Dataset
# Solo dataset reali e fortemente sbilanciati, presi da due raccolte pubbliche:
# - il benchmark di imbalanced-learn (``imblearn.datasets.fetch_datasets``,
#   Ding 2011), usato negli articoli sul learning sbilanciato;
# - OpenML (``sklearn.datasets.fetch_openml``), che conserva le colonne
#   categoriche con i loro valori originali.

def _binary_benchmark(name):
    """Dataset binario del benchmark di imbalanced-learn, con classi 0/1."""
    from imblearn.datasets import fetch_datasets

    d = fetch_datasets(filter_data=(name,))[name]
    return d.data.astype(float), (d.target == 1).astype(int), []


def load_abalone_19():
    """Abalone (UCI): l'abalone ha 19 anelli (età) contro tutti gli altri.

    Si usa la versione OpenML perché tiene ``Sex`` (M, F, I = immaturo) come
    colonna categorica; nel benchmark di imbalanced-learn è già in one-hot.
    """
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=183, as_frame=True).frame
    y = (df.pop("Class_number_of_rings").astype(int) == 19).astype(int).to_numpy()
    X = df.astype(object).to_numpy()
    return X, y, [df.columns.get_loc("Sex")]


def load_page_blocks():
    """Page Blocks (UCI): blocchi di pagine scansionate, 5 classi.

    1 testo, 2 riga orizzontale, 3 immagine, 4 riga verticale, 5 grafico.
    """
    from sklearn.datasets import fetch_openml

    b = fetch_openml(data_id=30, as_frame=True)
    return b.data.to_numpy(dtype=float), b.target.astype(int).to_numpy(), []


BANK_COLUMNS = {
    "V1": "age", "V2": "job", "V3": "marital", "V4": "education", "V5": "default",
    "V6": "balance", "V7": "housing", "V8": "loan", "V9": "contact", "V10": "day",
    "V11": "month", "V12": "duration", "V13": "campaign", "V14": "pdays",
    "V15": "previous", "V16": "poutcome",
}


def load_bank_marketing():
    """Bank Marketing (UCI, Moro et al. 2014): il cliente sottoscrive il deposito?

    9 colonne categoriche su 15. ``duration`` (durata della telefonata) viene
    tolta: si conosce solo a chiamata finita, quando l'esito è già noto, e
    usarla gonfierebbe i risultati (lo raccomandano gli autori del dataset).
    """
    from sklearn.datasets import fetch_openml

    b = fetch_openml(data_id=1461, as_frame=True)
    df = b.data.rename(columns=BANK_COLUMNS).drop(columns="duration")
    y = (b.target == "2").astype(int).to_numpy()  # "2" = sì
    cat_cols = [j for j, c in enumerate(df.columns) if df[c].dtype == "category"]
    return df.astype(object).to_numpy(), y, cat_cols


def load_datasets():
    """Restituisce {nome: (X, y, colonne_categoriche)}."""
    return {
        "satimage": _binary_benchmark("satimage"),
        "mammography": _binary_benchmark("mammography"),
        "abalone 19": load_abalone_19(),
        "page blocks": load_page_blocks(),
        "bank marketing": load_bank_marketing(),
    }


def describe(y):
    classes, counts = np.unique(y, return_counts=True)
    return ", ".join(f"{c}: {n}" for c, n in zip(classes, counts))


# Campionatori

def make_samplers(seed=SEED):
    """I metodi di imbalanced-learn confrontati, con i parametri di default."""
    return {
        "nessuno": None,
        "ROS": RandomOverSampler(random_state=seed),
        "SMOTE": SMOTE(random_state=seed),
        "Borderline-SMOTE": BorderlineSMOTE(random_state=seed),
        "ADASYN": ADASYN(random_state=seed),
        "RUS": RandomUnderSampler(random_state=seed),
        "NearMiss-1": NearMiss(version=1),
        "Tomek": TomekLinks(),
        "ENN": EditedNearestNeighbours(),
        "SMOTE+ENN": SMOTEENN(random_state=seed),
        "SMOTE+Tomek": SMOTETomek(random_state=seed),
    }


# Valutazione
def _preprocessor(cat_cols, n_features):
    num_cols = [j for j in range(n_features) if j not in cat_cols]
    if not cat_cols:
        return StandardScaler()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def _scores(y_true, y_pred, proba, classes):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "gmean": geometric_mean_score(y_true, y_pred),
    }
    if len(classes) == 2:
        out["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
    else:
        out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)
    return out


def evaluate(X, y, cat_cols, sampler, clf, n_splits=5, n_repeats=2, smotenc=False):
    """Cross-validation stratificata ripetuta con ricampionamento nel fold.

    Con ``smotenc=True`` il ricampionamento avviene PRIMA del one-hot:
    si standardizzano le continue, si applica SMOTENC sulle colonne grezze e
    solo dopo si codificano le categoriche.

    Restituisce una lista di dict (una riga per fold) con le metriche, i
    tempi e il numero di valori categorici "inventati" (non 0/1 nel one-hot).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    classes = np.unique(y)
    num_cols = [j for j in range(X.shape[1]) if j not in cat_cols]
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        t_resample = 0.0
        if smotenc:
            scaler = StandardScaler().fit(X_tr[:, num_cols].astype(float))
            X_tr, X_te = X_tr.copy(), X_te.copy()
            X_tr[:, num_cols] = scaler.transform(X_tr[:, num_cols].astype(float))
            X_te[:, num_cols] = scaler.transform(X_te[:, num_cols].astype(float))
            t0 = time.perf_counter()
            X_tr, y_tr = clone(sampler).fit_resample(X_tr, y_tr)
            t_resample = time.perf_counter() - t0
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc.fit(X_tr[:, cat_cols])
            X_tr = np.hstack([X_tr[:, num_cols].astype(float), enc.transform(X_tr[:, cat_cols])])
            X_te = np.hstack([X_te[:, num_cols].astype(float), enc.transform(X_te[:, cat_cols])])
            n_bad = 0
        else:
            pre = _preprocessor(cat_cols, X.shape[1]).fit(X_tr)
            X_tr, X_te = pre.transform(X_tr), pre.transform(X_te)
            n_before = X_tr.shape[0]
            if sampler is not None:
                t0 = time.perf_counter()
                X_tr, y_tr = clone(sampler).fit_resample(X_tr, y_tr)
                t_resample = time.perf_counter() - t0
            n_bad = 0
            if cat_cols and X_tr.shape[0] > n_before:
                ohe = X_tr[n_before:, len(num_cols):]
                n_bad = int(np.sum(~np.isclose(ohe, 0) & ~np.isclose(ohe, 1)))

        model = clone(clf).fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        proba = model.predict_proba(X_te)
        row = _scores(y_te, y_pred, proba, classes)
        row.update(
            fold=fold,
            n_train_resampled=X_tr.shape[0],
            invalid_categorical_values=n_bad,
            resample_seconds=t_resample,
        )
        rows.append(row)
    return rows
