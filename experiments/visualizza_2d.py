"""Figura didattica: cosa fa ogni metodo su un dataset 2D sbilanciato.

Uso:  python experiments/visualizza_2d.py   ->  results/visualizzazione_2d.png
"""

import sys
from pathlib import Path

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)

sys.path.insert(0, str(Path(__file__).parent))
from plots import scatter_grid  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    # Due nuvole gaussiane parzialmente sovrapposte: 450 vs 50 campioni.
    rng = np.random.RandomState(7)
    X = np.vstack([
        rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 1.0]], size=450),
        rng.multivariate_normal([1.6, 1.4], [[0.35, 0], [0, 0.35]], size=50),
    ])
    y = np.repeat([0, 1], [450, 50])
    samplers = {
        "ROS (duplicati)": RandomOverSampler(random_state=0),
        "ROS con shrinkage": RandomOverSampler(random_state=0, shrinkage=0.5),
        "SMOTE": SMOTE(random_state=0),
        "Borderline-SMOTE": BorderlineSMOTE(random_state=0),
        "ADASYN": ADASYN(random_state=0),
        "RUS": RandomUnderSampler(random_state=0),
        "NearMiss-1": NearMiss(version=1),
        "NearMiss-3": NearMiss(version=3),
        "Tomek links": TomekLinks(),
        "ENN": EditedNearestNeighbours(),
        "SMOTE+ENN": SMOTEENN(random_state=0),
    }
    panels = [("Originale", X, y, len(y))]
    for name, s in samplers.items():
        Xr, yr = s.fit_resample(X, y)
        if isinstance(s, SMOTEENN):
            # ENN tiene gli indici in ordine: i sintetici restano in coda.
            n_orig = int((s.enn_.sample_indices_ < len(y)).sum())
        elif len(yr) > len(y):
            n_orig = len(y)  # oversampling: i sintetici sono accodati
        else:
            n_orig = len(yr)  # undersampling: solo campioni reali
        panels.append((name, Xr, yr, n_orig))
    scatter_grid(panels, RESULTS / "visualizzazione_2d.png", ncols=4)
    print(f"Salvato {RESULTS / 'visualizzazione_2d.png'}")


if __name__ == "__main__":
    main()
