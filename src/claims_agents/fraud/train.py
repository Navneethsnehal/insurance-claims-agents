"""Train the synthetic-claim gradient boosting model.

The generator is seeded. Running this module twice on the same version of
the code rewrites identical CSVs and a model with the same held-out metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from claims_agents.fraud.features import FEATURE_NAMES

TRAIN_SEED = 42
N_TRAIN = 4000
N_TEST = 1200
FRAUD_RATE = 0.18
LABEL_NOISE = 0.05


def _sample_row(rng: np.random.Generator, fraud: bool) -> list[float]:
    if fraud:
        amount = float(rng.uniform(6_000, 80_000))
        age = int(rng.integers(1, 120))
        prior = int(rng.integers(1, 8))
        days = int(rng.integers(0, 90))
        late = int(rng.random() < 0.60)
        attorney = int(rng.random() < 0.40)
        inconsistent = int(rng.random() < 0.50)
        police = int(rng.random() < 0.40)
        ratio = float(rng.uniform(0.70, 1.50))
        siu = int(rng.random() < 0.40)
        photos = int(rng.integers(0, 4))
        witnesses = int(rng.integers(0, 2))
        provider = int(rng.integers(2, 9))
    else:
        large = rng.random() < 0.25
        amount = float(rng.uniform(15_000, 50_000) if large else rng.uniform(400, 12_000))
        age = int(rng.integers(180, 2_500))
        prior = int(rng.integers(0, 2))
        days = int(rng.integers(0, 12))
        late = int(rng.random() < 0.05)
        attorney = int(rng.random() < 0.04)
        inconsistent = 0
        police = int(rng.random() < 0.85)
        ratio = float(rng.uniform(0.05, 0.60))
        siu = 0
        photos = int(rng.integers(3, 12))
        witnesses = int(rng.integers(0, 3))
        provider = int(rng.integers(0, 2))
    new_high = 1.0 if age < 30 and amount >= 15_000 else 0.0
    return [
        amount,
        float(age),
        float(prior),
        float(days),
        float(late),
        float(attorney),
        float(inconsistent),
        float(police),
        ratio,
        new_high,
        float(siu),
        float(photos),
        float(witnesses),
        float(provider),
    ]


def synthesize(
    n_rows: int,
    seed: int,
    fraud_rate: float = FRAUD_RATE,
    label_noise: float = LABEL_NOISE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    for _ in range(n_rows):
        fraud = bool(rng.random() < fraud_rate)
        rows.append(_sample_row(rng, fraud))
        observed = (not fraud) if rng.random() < label_noise else fraud
        labels.append(int(observed))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(labels))
    if k == 0:
        return 0.0
    order = np.argsort(-scores)[:k]
    return float(labels[order].mean())


def _write_csv(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*FEATURE_NAMES, "is_fraud"])
        for row, label in zip(features, labels, strict=True):
            writer.writerow([*(float(value) for value in row), int(label)])


def train_and_save(
    model_path: Path,
    train_csv: Path,
    test_csv: Path,
    *,
    seed: int = TRAIN_SEED,
) -> dict[str, float]:
    x_train, y_train = synthesize(N_TRAIN, seed)
    x_test, y_test = synthesize(N_TEST, seed + 1)
    _write_csv(train_csv, x_train, y_train)
    _write_csv(test_csv, x_test, y_test)
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.06,
        max_depth=3,
        subsample=0.9,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "precision_at_100": precision_at_k(y_test, np.asarray(scores, dtype=np.float64), 100),
        "test_rows": float(len(y_test)),
        "test_fraud_rate": float(y_test.mean()),
    }
    baseline = np.median(x_train, axis=0)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(FEATURE_NAMES),
            "baseline": baseline.tolist(),
            "metrics": metrics,
            "seed": seed,
        },
        model_path,
    )
    model_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def load_bundle(model_path: Path, train_csv: Path, test_csv: Path) -> dict[str, object]:
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
        except Exception:
            bundle = None
        names = list(bundle.get("feature_names", [])) if isinstance(bundle, dict) else []
        if names == list(FEATURE_NAMES) and isinstance(bundle, dict):
            return bundle
    train_and_save(model_path, train_csv, test_csv)
    loaded = joblib.load(model_path)
    if not isinstance(loaded, dict):
        raise TypeError("fraud model bundle must be a dict")
    return loaded


def main() -> None:
    from claims_agents.config import get_settings

    settings = get_settings()
    metrics = train_and_save(
        settings.fraud_model_path,
        settings.fraud_train_path,
        settings.fraud_test_path,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
