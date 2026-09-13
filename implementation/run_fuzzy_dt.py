"""Fuzzy Decision Tree + Fuzzy Forest model training.

Key improvements over previous version:
- Enriched feature set (UDP density, byte density, packet length stats)
- Additional fuzzy rules covering UDP flood and volumetric DDoS patterns
- class_weight='balanced_subsample' for Random Forest to handle 85:15 class imbalance
- Paths updated to new folder structure (data/, reports/)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from shared_preprocessing import build_clean_dataset


FUZZY_ATTRS = [
    "syn_ack_ratio",
    "half_open_conn_count",
    "same_src_ip_freq",
    "syn_packet_density",
    "udp_packet_density",
    "total_packet_density",
    "byte_density",
    "unique_dst_port_count",
    "avg_time_between_syns",
]


MEMBERSHIP_PARAMS = {
    "syn_ack_ratio":        {"LOW": [0.0, 0.0, 0.3], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.5, 1.0, 1.0]},
    "half_open_conn_count": {"LOW": [0.0, 0.0, 0.3], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.5, 1.0, 1.0]},
    "same_src_ip_freq":     {"LOW": [0.0, 0.0, 0.25], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.6, 1.0, 1.0]},
    "syn_packet_density":   {"LOW": [0.0, 0.0, 0.25], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.5, 1.0, 1.0]},
    "udp_packet_density":   {"LOW": [0.0, 0.0, 0.25], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.5, 1.0, 1.0]},
    "total_packet_density": {"LOW": [0.0, 0.0, 0.25], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.5, 1.0, 1.0]},
    "byte_density":         {"LOW": [0.0, 0.0, 0.25], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.5, 1.0, 1.0]},
    "unique_dst_port_count":{"LOW": [0.0, 0.0, 0.3], "MEDIUM": [0.2, 0.5, 0.8], "HIGH": [0.7, 1.0, 1.0]},
    "avg_time_between_syns":{"LOW": [0.0, 0.0, 0.25], "MEDIUM": [0.15, 0.45, 0.7], "HIGH": [0.6, 1.0, 1.0]},
}


def _trimf(x: np.ndarray, abc: list[float]) -> np.ndarray:
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    if b > a:
        idx = (x >= a) & (x <= b)
        y[idx] = (x[idx] - a) / (b - a)
    else:
        y[x == a] = 1.0
    if c > b:
        idx = (x >= b) & (x <= c)
        y[idx] = np.maximum(y[idx], (c - x[idx]) / (c - b))
    else:
        y[x == c] = 1.0
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def _normalize_with_train_quantiles(
    df: pd.DataFrame,
    train_mask: pd.Series,
    cols: list[str],
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=df.index)
    bounds: dict = {}
    for col in cols:
        train_values = df.loc[train_mask, col].fillna(0)
        lo = float(train_values.quantile(low_q))
        hi = float(train_values.quantile(high_q))
        span = hi - lo
        clipped = df[col].fillna(0).clip(lower=lo, upper=hi)
        out[col] = ((clipped - lo) / span).clip(0, 1) if span > 0 else 0.0
        bounds[col] = (lo, hi)
    return out, bounds


def _threshold_tune(
    y_true: pd.Series,
    y_score: np.ndarray,
    min_recall: float = 0.35,
) -> float:
    """Tune threshold to maximize F1 subject to minimum recall floor."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        # Balanced combination: Accuracy primary (0.55), F1 (0.30), Balanced Accuracy (0.15)
        score = (
            0.55 * accuracy_score(y_true, pred)
            + 0.30 * f1_score(y_true, pred, zero_division=0)
            + 0.15 * balanced_accuracy_score(y_true, pred)
        )
        if score > best_score:
            best_score = score
            best_t = float(t)
    return float(best_t) if best_score >= 0 else 0.5


def _metric_bundle(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
    }


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

    fuzzy_norm, clip_bounds = _normalize_with_train_quantiles(df, train_mask, FUZZY_ATTRS)

    # Build fuzzy membership scores
    fuzzy_scores = pd.DataFrame(index=df.index)
    for attr in FUZZY_ATTRS:
        values = fuzzy_norm[attr].to_numpy(dtype=float)
        for level in ["LOW", "MEDIUM", "HIGH"]:
            fuzzy_scores[f"{attr}_{level}"] = _trimf(values, MEMBERSHIP_PARAMS[attr][level])

    # Build fuzzy rules covering SYN flood, UDP flood, volumetric and stealth patterns
    rule_results = pd.DataFrame(index=df.index)
    rule_results["rule1_syn_ack_halfopen"]  = np.minimum(fuzzy_scores["syn_ack_ratio_HIGH"], fuzzy_scores["half_open_conn_count_HIGH"])
    rule_results["rule2_density_burst"]     = np.minimum(fuzzy_scores["syn_packet_density_HIGH"], fuzzy_scores["avg_time_between_syns_LOW"])
    rule_results["rule3_spoofed_ip"]        = np.minimum(fuzzy_scores["same_src_ip_freq_LOW"], fuzzy_scores["syn_ack_ratio_HIGH"])
    rule_results["rule4_retransmit_halfopen"]= np.minimum(fuzzy_scores["syn_packet_density_HIGH"], fuzzy_scores["half_open_conn_count_HIGH"])
    rule_results["rule5_targeted_port"]     = np.minimum(fuzzy_scores["unique_dst_port_count_LOW"], fuzzy_scores["syn_packet_density_HIGH"])
    rule_results["rule6_udp_density_burst"] = np.minimum(fuzzy_scores["udp_packet_density_HIGH"], fuzzy_scores["total_packet_density_HIGH"])
    rule_results["rule7_high_byte_flood"]   = np.minimum(fuzzy_scores["byte_density_HIGH"], fuzzy_scores["total_packet_density_HIGH"])
    rule_results["rule8_stealth_flood"]     = np.minimum.reduce([
        fuzzy_scores["udp_packet_density_MEDIUM"],
        fuzzy_scores["syn_ack_ratio_MEDIUM"],
        fuzzy_scores["total_packet_density_MEDIUM"],
    ])

    RULE_COLS = [f"rule{i}_{n}" for i, n in enumerate([
        "syn_ack_halfopen", "density_burst", "spoofed_ip", "retransmit_halfopen",
        "targeted_port", "udp_density_burst", "high_byte_flood", "stealth_flood"
    ], 1)]

    weights = np.array([0.16, 0.15, 0.12, 0.10, 0.10, 0.15, 0.12, 0.10], dtype=float)
    rule_mat = rule_results[RULE_COLS].to_numpy(dtype=float)
    rule_results["fuzzy_confidence"] = 0.55 * (rule_mat @ weights) + 0.45 * rule_mat.max(axis=1)

    model_df = pd.concat([df, fuzzy_scores, rule_results], axis=1)

    original_features = [
        "packet_length", "time_delta", "is_syn_only",
        "syn_ack_ratio", "half_open_conn_count", "same_src_ip_freq",
        "syn_packet_density", "udp_packet_density", "total_packet_density", "byte_density",
        "retransmission_rate", "unique_dst_port_count", "avg_time_between_syns",
        "pkt_len_mean", "pkt_len_std", "udp_ratio", "syn_ratio",
    ]
    membership_features = [f"{a}_{l}" for a in FUZZY_ATTRS for l in ["LOW", "MEDIUM", "HIGH"]]
    rule_features = RULE_COLS + ["fuzzy_confidence"]
    fuzzy_dt_features = original_features + membership_features + rule_features

    X_train = model_df.loc[train_mask, fuzzy_dt_features].fillna(0)
    X_test  = model_df.loc[test_mask,  fuzzy_dt_features].fillna(0)
    y_train = model_df.loc[train_mask, "attack"].astype(int)
    y_test  = model_df.loc[test_mask,  "attack"].astype(int)

    # ── Fuzzy Decision Tree (Single tree — lightweight, fast inference) ──────
    fuzzy_dt = DecisionTreeClassifier(
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 2.5},
        random_state=42,
    )
    fuzzy_dt.fit(X_train, y_train)
    dt_proba_train = fuzzy_dt.predict_proba(X_train)[:, 1]
    dt_proba_test  = fuzzy_dt.predict_proba(X_test)[:, 1]
    th_dt = _threshold_tune(y_train, dt_proba_train, min_recall=0.35)
    y_pred_dt = (dt_proba_test >= th_dt).astype(int)
    fuzzy_dt_metrics = _metric_bundle(y_test, y_pred_dt, dt_proba_test)

    # ── Fuzzy Forest (Compact ensemble: 30 trees, 2.7 MB instead of 44 MB) ───
    forest_model = RandomForestClassifier(
        n_estimators=30,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 2.5},
        n_jobs=-1,
        random_state=42,
    )
    forest_model.fit(X_train, y_train)
    forest_proba_train = forest_model.predict_proba(X_train)[:, 1]
    forest_proba_test  = forest_model.predict_proba(X_test)[:, 1]
    th_forest = _threshold_tune(y_train, forest_proba_train, min_recall=0.35)
    y_pred_forest = (forest_proba_test >= th_forest).astype(int)
    fuzzy_forest_metrics = _metric_bundle(y_test, y_pred_forest, forest_proba_test)

    # ── Save deployable artifact ──────────────────────────────────────────────
    artifact = {
        "artifact_version": 3,
        "model": forest_model,
        "fuzzy_dt_single": fuzzy_dt,
        "feature_order": fuzzy_dt_features,
        "fuzzy_attributes": FUZZY_ATTRS,
        "membership_params": MEMBERSHIP_PARAMS,
        "rule_weights": weights.tolist(),
        "clip_bounds": {k: [v[0], v[1]] for k, v in clip_bounds.items()},
        "threshold": float(th_forest),
        "fuzzy_dt_threshold": float(th_dt),
        "window": 50,
    }
    joblib.dump(artifact, "fuzzy_forest.pkl")

    report = {
        "dataset": str(clean_path),
        "artifact": "fuzzy_forest.pkl",
        "feature_count": len(fuzzy_dt_features),
        "fuzzy_dt_single_metrics": fuzzy_dt_metrics,
        "fuzzy_forest_metrics": fuzzy_forest_metrics,
        "classification_report_forest": classification_report(
            y_test, y_pred_forest, target_names=["Normal", "Attack"], output_dict=True
        ),
        "classification_report_dt": classification_report(
            y_test, y_pred_dt, target_names=["Normal", "Attack"], output_dict=True
        ),
    }
    Path("reports").mkdir(exist_ok=True)
    with Path("reports/fuzzy_dt_results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Fuzzy model training complete.")
    print(f"  Fuzzy DT  (single tree): {fuzzy_dt_metrics}")
    print(f"  Fuzzy Forest:            {fuzzy_forest_metrics}")
    print("Saved: fuzzy_forest.pkl | reports/fuzzy_dt_results.json")


if __name__ == "__main__":
    main()
