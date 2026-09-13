"""Normal Decision Tree baseline — GridSearchCV tuning, corrected paths.

Trained WITHOUT any fuzzy features so the comparison is honest.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


FEATURES = [
    "packet_length", "time_delta", "is_syn_only",
    "syn_ack_ratio", "half_open_conn_count", "same_src_ip_freq",
    "syn_packet_density", "udp_packet_density", "total_packet_density",
    "byte_density", "retransmission_rate", "unique_dst_port_count",
    "avg_time_between_syns", "pkt_len_mean", "pkt_len_std",
    "udp_ratio", "syn_ratio",
]


def _threshold_tune(y_true, y_score, min_recall: float = 0.35) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        score = (
            0.60 * f1_score(y_true, pred, zero_division=0)
            + 0.25 * accuracy_score(y_true, pred)
            + 0.15 * balanced_accuracy_score(y_true, pred)
        )
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t if best_score >= 0 else 0.5


def main() -> None:
    clean_path = Path("data/syn_udp_flood_attack_data_clean.csv")
    if not clean_path.exists():
        build_clean_dataset(
            input_csv="data/syn_udp_flood_attack_data_labeled.csv",
            output_csv=str(clean_path),
            output_report="reports/clean_dataset_report.json",
        )

    df = pd.read_csv(clean_path)
    train_mask = df["split"] == "train"
    test_mask  = df["split"] == "test"

    X_train = df.loc[train_mask, FEATURES].fillna(0)
    X_test  = df.loc[test_mask,  FEATURES].fillna(0)
    y_train = df.loc[train_mask, "attack"].astype(int)
    y_test  = df.loc[test_mask,  "attack"].astype(int)

    param_grid = {
        "max_depth": [8, 10, 12, 15, 18],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test  = model.predict_proba(X_test)[:, 1]
    threshold   = _threshold_tune(y_train, proba_train)
    y_pred      = (proba_test >= threshold).astype(int)

    metrics = {
        "accuracy":          float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision":         float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_test, y_pred, zero_division=0)),
        "auc":               float(roc_auc_score(y_test, proba_test)),
    }

    report = {
        "model_type": "Normal_DT",
        "best_params": grid.best_params_,
        "feature_count": len(FEATURES),
        "threshold": float(threshold),
        "metrics": metrics,
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Normal", "Attack"],
            output_dict=True,
        ),
    }
    Path("reports").mkdir(exist_ok=True)
    with Path("reports/normal_dt_results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Normal DT training complete.")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Metrics:     {metrics}")
    print("Saved: reports/normal_dt_results.json")


if __name__ == "__main__":
    main()
