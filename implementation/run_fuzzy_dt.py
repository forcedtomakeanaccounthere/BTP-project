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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from shared_preprocessing import build_clean_dataset


FUZZY_ATTRS = [
    "syn_ack_ratio",
    "half_open_conn_count",
    "same_src_ip_freq",
    "syn_packet_density",
    "retransmission_rate",
    "unique_dst_port_count",
    "avg_time_between_syns",
]


MEMBERSHIP_PARAMS = {
    "syn_ack_ratio": {
        "LOW": [0.0, 0.0, 0.3],
        "MEDIUM": [0.15, 0.45, 0.7],
        "HIGH": [0.5, 1.0, 1.0],
    },
    "half_open_conn_count": {
        "LOW": [0.0, 0.0, 0.3],
        "MEDIUM": [0.15, 0.45, 0.7],
        "HIGH": [0.5, 1.0, 1.0],
    },
    "same_src_ip_freq": {
        "LOW": [0.0, 0.0, 0.25],
        "MEDIUM": [0.15, 0.45, 0.7],
        "HIGH": [0.6, 1.0, 1.0],
    },
    "syn_packet_density": {
        "LOW": [0.0, 0.0, 0.25],
        "MEDIUM": [0.15, 0.45, 0.7],
        "HIGH": [0.5, 1.0, 1.0],
    },
    "retransmission_rate": {
        "LOW": [0.0, 0.0, 0.2],
        "MEDIUM": [0.1, 0.4, 0.7],
        "HIGH": [0.5, 1.0, 1.0],
    },
    "unique_dst_port_count": {
        "LOW": [0.0, 0.0, 0.3],
        "MEDIUM": [0.2, 0.5, 0.8],
        "HIGH": [0.7, 1.0, 1.0],
    },
    "avg_time_between_syns": {
        "LOW": [0.0, 0.0, 0.25],
        "MEDIUM": [0.15, 0.45, 0.7],
        "HIGH": [0.6, 1.0, 1.0],
    },
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
    low_q: float = 0.02,
    high_q: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    out = pd.DataFrame(index=df.index)
    bounds: dict[str, tuple[float, float]] = {}

    for col in cols:
        train_values = df.loc[train_mask, col].fillna(0)
        lo = float(train_values.quantile(low_q))
        hi = float(train_values.quantile(high_q))
        span = hi - lo

        clipped = df[col].fillna(0).clip(lower=lo, upper=hi)
        if span > 0:
            out[col] = ((clipped - lo) / span).clip(0, 1)
        else:
            out[col] = 0.0

        bounds[col] = (lo, hi)

    return out, bounds


def _threshold_tune(
    y_true: pd.Series,
    y_score: np.ndarray,
    profile: str = "accuracy",
    min_recall: float = 0.45,
) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        if profile == "detection":
            score = (
                0.70 * f1_score(y_true, pred, zero_division=0)
                + 0.20 * balanced_accuracy_score(y_true, pred)
                + 0.10 * accuracy_score(y_true, pred)
            )
        else:
            score = (
                0.70 * accuracy_score(y_true, pred)
                + 0.20 * f1_score(y_true, pred, zero_division=0)
                + 0.10 * balanced_accuracy_score(y_true, pred)
            )
        if score > best_score:
            best_score = score
            best_t = float(t)
    if best_score < 0:
        return 0.5
    return best_t


def _metric_bundle(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
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
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    fuzzy_norm, clip_bounds = _normalize_with_train_quantiles(df, train_mask, FUZZY_ATTRS)

    fuzzy_scores = pd.DataFrame(index=df.index)
    for attr in FUZZY_ATTRS:
        values = fuzzy_norm[attr].to_numpy(dtype=float)
        for level in ["LOW", "MEDIUM", "HIGH"]:
            fuzzy_scores[f"{attr}_{level}"] = _trimf(values, MEMBERSHIP_PARAMS[attr][level])

    rule_results = pd.DataFrame(index=df.index)
    rule_results["rule1_syn_ack_halfopen"] = np.minimum(
        fuzzy_scores["syn_ack_ratio_HIGH"], fuzzy_scores["half_open_conn_count_HIGH"]
    )
    rule_results["rule2_density_burst"] = np.minimum(
        fuzzy_scores["syn_packet_density_HIGH"], fuzzy_scores["avg_time_between_syns_LOW"]
    )
    rule_results["rule3_spoofed_ip"] = np.minimum(
        fuzzy_scores["same_src_ip_freq_LOW"], fuzzy_scores["syn_ack_ratio_HIGH"]
    )
    rule_results["rule4_retransmit_halfopen"] = np.minimum(
        fuzzy_scores["retransmission_rate_HIGH"], fuzzy_scores["half_open_conn_count_HIGH"]
    )
    rule_results["rule5_targeted_port"] = np.minimum(
        fuzzy_scores["unique_dst_port_count_LOW"], fuzzy_scores["syn_packet_density_HIGH"]
    )
    # Low-level/stealth flood behavior: medium SYN pressure with short inter-arrival times.
    rule_results["rule6_low_rate_flood"] = np.minimum.reduce(
        [
            fuzzy_scores["syn_ack_ratio_MEDIUM"],
            fuzzy_scores["half_open_conn_count_MEDIUM"],
            fuzzy_scores["avg_time_between_syns_LOW"],
        ]
    )
    # Congestion-like suspicious behavior: medium burst density + medium half-open accumulation.
    rule_results["rule7_congestion_suspicion"] = np.minimum(
        fuzzy_scores["syn_packet_density_MEDIUM"],
        fuzzy_scores["half_open_conn_count_MEDIUM"],
    )
    # Multi-attribute weak attack pattern: spoofing tendency + targeted destination behavior.
    rule_results["rule8_weak_pattern"] = np.minimum.reduce(
        [
            fuzzy_scores["same_src_ip_freq_MEDIUM"],
            fuzzy_scores["syn_ack_ratio_MEDIUM"],
            fuzzy_scores["unique_dst_port_count_LOW"],
        ]
    )

    weights = np.array([0.20, 0.18, 0.14, 0.10, 0.10, 0.12, 0.08, 0.08], dtype=float)
    rule_mat = rule_results[
        [
            "rule1_syn_ack_halfopen",
            "rule2_density_burst",
            "rule3_spoofed_ip",
            "rule4_retransmit_halfopen",
            "rule5_targeted_port",
            "rule6_low_rate_flood",
            "rule7_congestion_suspicion",
            "rule8_weak_pattern",
        ]
    ].to_numpy(dtype=float)

    combined = rule_mat @ weights
    max_rule = rule_mat.max(axis=1)
    rule_results["fuzzy_confidence"] = 0.55 * combined + 0.45 * max_rule

    model_df = pd.concat([df, fuzzy_scores, rule_results], axis=1)

    original_features = [
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
    ]

    rule_features = [
        "rule1_syn_ack_halfopen",
        "rule2_density_burst",
        "rule3_spoofed_ip",
        "rule4_retransmit_halfopen",
        "rule5_targeted_port",
        "rule6_low_rate_flood",
        "rule7_congestion_suspicion",
        "rule8_weak_pattern",
        "fuzzy_confidence",
    ]

    membership_features = [
        f"{attr}_{level}"
        for attr in FUZZY_ATTRS
        for level in ["LOW", "MEDIUM", "HIGH"]
    ]

    fuzzy_dt_features = original_features + membership_features + rule_features

    X_train = model_df.loc[train_mask, fuzzy_dt_features].fillna(0)
    X_test = model_df.loc[test_mask, fuzzy_dt_features].fillna(0)
    y_train = model_df.loc[train_mask, "attack"].astype(int)
    y_test = model_df.loc[test_mask, "attack"].astype(int)

    fuzzy_conf_train = model_df.loc[train_mask, "fuzzy_confidence"].to_numpy(dtype=float)
    fuzzy_conf_test = model_df.loc[test_mask, "fuzzy_confidence"].to_numpy(dtype=float)

    th_fuzzy_acc = _threshold_tune(y_train, fuzzy_conf_train, profile="accuracy", min_recall=0.45)
    y_pred_fuzzy_acc = (fuzzy_conf_test >= th_fuzzy_acc).astype(int)
    fuzzy_threshold_metrics_acc = _metric_bundle(y_test, y_pred_fuzzy_acc, fuzzy_conf_test)

    th_fuzzy_det = _threshold_tune(y_train, fuzzy_conf_train, profile="detection", min_recall=0.55)
    y_pred_fuzzy_det = (fuzzy_conf_test >= th_fuzzy_det).astype(int)
    fuzzy_threshold_metrics_det = _metric_bundle(y_test, y_pred_fuzzy_det, fuzzy_conf_test)

    model = DecisionTreeClassifier(random_state=42)

    param_grid = {
        "criterion": ["gini"],
        "max_depth": [5, 6, 8],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "ccp_alpha": [0.0, 0.0005],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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
    proba_train = best_model.predict_proba(X_train)[:, 1]
    proba_test = best_model.predict_proba(X_test)[:, 1]

    th_dt_acc = _threshold_tune(y_train, proba_train, profile="accuracy", min_recall=0.45)
    y_pred_dt_acc = (proba_test >= th_dt_acc).astype(int)
    fuzzy_dt_metrics_acc = _metric_bundle(y_test, y_pred_dt_acc, proba_test)

    th_dt_det = _threshold_tune(y_train, proba_train, profile="detection", min_recall=0.55)
    y_pred_dt_det = (proba_test >= th_dt_det).astype(int)
    fuzzy_dt_metrics_det = _metric_bundle(y_test, y_pred_dt_det, proba_test)

    forest_model = RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    forest_model.fit(X_train, y_train)

    forest_proba_train = forest_model.predict_proba(X_train)[:, 1]
    forest_proba_test = forest_model.predict_proba(X_test)[:, 1]

    # Balanced profile tuned to keep high recall while preserving strong accuracy and precision.
    th_forest_bal = _threshold_tune(
        y_train,
        forest_proba_train,
        profile="accuracy",
        min_recall=0.54,
    )
    y_pred_forest_bal = (forest_proba_test >= th_forest_bal).astype(int)
    fuzzy_forest_metrics_bal = _metric_bundle(y_test, y_pred_forest_bal, forest_proba_test)

    th_forest_det = _threshold_tune(
        y_train,
        forest_proba_train,
        profile="detection",
        min_recall=0.60,
    )
    y_pred_forest_det = (forest_proba_test >= th_forest_det).astype(int)
    fuzzy_forest_metrics_det = _metric_bundle(y_test, y_pred_forest_det, forest_proba_test)

    report = {
        "dataset": str(clean_path),
        "recommended_profile": "fuzzy_forest_balanced_profile",
        "feature_count": len(fuzzy_dt_features),
        "clip_bounds": {k: [v[0], v[1]] for k, v in clip_bounds.items()},
        "fuzzy_threshold_accuracy_profile": {
            "threshold": float(th_fuzzy_acc),
            "test_metrics": fuzzy_threshold_metrics_acc,
        },
        "fuzzy_threshold_detection_profile": {
            "threshold": float(th_fuzzy_det),
            "test_metrics": fuzzy_threshold_metrics_det,
        },
        "fuzzy_dt_accuracy_profile": {
            "best_params": search.best_params_,
            "cv_best_roc_auc": float(search.best_score_),
            "threshold": float(th_dt_acc),
            "test_metrics": fuzzy_dt_metrics_acc,
            "classification_report": classification_report(
                y_test, y_pred_dt_acc, target_names=["Normal", "Attack"], output_dict=True
            ),
        },
        "fuzzy_dt_detection_profile": {
            "threshold": float(th_dt_det),
            "test_metrics": fuzzy_dt_metrics_det,
            "classification_report": classification_report(
                y_test, y_pred_dt_det, target_names=["Normal", "Attack"], output_dict=True
            ),
        },
        "fuzzy_forest_balanced_profile": {
            "model_params": {
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
            },
            "threshold": float(th_forest_bal),
            "test_metrics": fuzzy_forest_metrics_bal,
            "classification_report": classification_report(
                y_test, y_pred_forest_bal, target_names=["Normal", "Attack"], output_dict=True
            ),
        },
        "fuzzy_forest_detection_profile": {
            "threshold": float(th_forest_det),
            "test_metrics": fuzzy_forest_metrics_det,
            "classification_report": classification_report(
                y_test, y_pred_forest_det, target_names=["Normal", "Attack"], output_dict=True
            ),
        },
    }

    with Path("fuzzy_dt_results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Fuzzy model training complete (fuzzy threshold + fuzzy DT + fuzzy forest profiles).")
    print(
        f"Fuzzy threshold (accuracy profile): "
        f"{ {k: round(v, 4) for k, v in fuzzy_threshold_metrics_acc.items()} }"
    )
    print(
        f"Fuzzy threshold (detection profile): "
        f"{ {k: round(v, 4) for k, v in fuzzy_threshold_metrics_det.items()} }"
    )
    print(
        f"Fuzzy DT (accuracy profile): "
        f"{ {k: round(v, 4) for k, v in fuzzy_dt_metrics_acc.items()} }"
    )
    print(
        f"Fuzzy DT (detection profile): "
        f"{ {k: round(v, 4) for k, v in fuzzy_dt_metrics_det.items()} }"
    )
    print(
        f"Fuzzy Forest (balanced profile): "
        f"{ {k: round(v, 4) for k, v in fuzzy_forest_metrics_bal.items()} }"
    )
    print(
        f"Fuzzy Forest (detection profile): "
        f"{ {k: round(v, 4) for k, v in fuzzy_forest_metrics_det.items()} }"
    )


if __name__ == "__main__":
    main()
