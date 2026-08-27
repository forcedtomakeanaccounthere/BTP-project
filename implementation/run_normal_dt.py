import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from shared_preprocessing import build_clean_dataset


def _threshold_tune(y_true: pd.Series, y_proba: np.ndarray, min_recall: float = 0.45) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        score = (
            0.70 * accuracy_score(y_true, pred)
            + 0.20 * f1_score(y_true, pred, zero_division=0)
            + 0.10 * balanced_accuracy_score(y_true, pred)
        )
        if score > best_score:
            best_score = score
            best_t = float(t)
    if best_score < 0:
        # Fallback when recall floor is too strict on a particular split.
        return 0.5
    return best_t


def _metric_bundle(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


def main() -> None:
    clean_path = Path("syn_udp_flood_attack_data_clean.csv")
    if not clean_path.exists():
        build_clean_dataset(
            input_csv="syn_udp_flood_attack_data_labeled.csv",
            output_csv=str(clean_path),
            output_report="clean_dataset_report.json",
        )

    df = pd.read_csv(clean_path)

    feature_cols = [
        "packet_length",
        "time_delta",
        "is_syn_only",
        "syn_ack_ratio",
        "half_open_conn_count",
        "same_src_ip_freq",
        "syn_packet_density",
        "retransmission_rate",
        "unique_dst_port_count",
        "avg_time_between_syns",
        "is_tcp",
        "is_udp",
    ]

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    X_test = df.loc[test_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, "attack"].astype(int)
    y_test = df.loc[test_mask, "attack"].astype(int)

    model = DecisionTreeClassifier(random_state=42)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 6, 8],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "ccp_alpha": [0.0, 0.0005],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    train_proba = best_model.predict_proba(X_train)[:, 1]
    test_proba = best_model.predict_proba(X_test)[:, 1]

    threshold = _threshold_tune(y_train, train_proba)
    y_pred = (test_proba >= threshold).astype(int)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_proba = np.zeros_like(dummy_pred, dtype=float)

    model_metrics = _metric_bundle(y_test, y_pred, test_proba)
    dummy_metrics = _metric_bundle(y_test, dummy_pred, dummy_proba)

    report = {
        "dataset": str(clean_path),
        "feature_count": len(feature_cols),
        "best_params": search.best_params_,
        "cv_best_f1": float(search.best_score_),
        "threshold": float(threshold),
        "normal_dt_test_metrics": model_metrics,
        "dummy_test_metrics": dummy_metrics,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Normal", "Attack"], output_dict=True
        ),
    }

    with Path("normal_dt_results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Normal DT training complete (GridSearchCV, no under-sampling).")
    print(f"Best CV F1: {search.best_score_:.4f}")
    print(f"Threshold: {threshold:.3f}")
    print("Test metrics (Normal DT):", {k: round(v, 4) for k, v in model_metrics.items()})
    print("Test metrics (Dummy):", {k: round(v, 4) for k, v in dummy_metrics.items()})


if __name__ == "__main__":
    main()
