"""Pi-friendly fast inference for the serialized fuzzy forest artifact.

Key fix: _fuzzy_features() is called ONCE per prediction, removing the
duplicate computation that was causing 26ms latency vs 0.06ms for plain DT.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_fuzzy_dt import _trimf


RULE_COLUMNS = [
    "rule1_syn_ack_halfopen",
    "rule2_density_burst",
    "rule3_spoofed_ip",
    "rule4_retransmit_halfopen",
    "rule5_targeted_port",
    "rule6_udp_density_burst",
    "rule7_high_byte_flood",
    "rule8_stealth_flood",
]


class FuzzyForestPredictor:
    def __init__(self, artifact_path: str = "fuzzy_forest.pkl") -> None:
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run run_fuzzy_dt.py on the laptop first."
            )
        self.artifact = joblib.load(path)
        self.model = self.artifact["model"]
        self.feature_order = self.artifact["feature_order"]
        self.fuzzy_attributes = self.artifact["fuzzy_attributes"]
        self.membership_params = self.artifact["membership_params"]
        self.rule_weights = np.asarray(self.artifact["rule_weights"], dtype=float)
        self.clip_bounds = self.artifact["clip_bounds"]
        self.threshold = float(self.artifact["threshold"])

    def _fuzzy_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute fuzzy membership scores + rule activations. Called ONCE per predict()."""
        result = features.copy()
        fuzzy_scores = pd.DataFrame(index=result.index)

        for attribute in self.fuzzy_attributes:
            values = result.get(attribute, pd.Series(0, index=result.index)).fillna(0).to_numpy(dtype=float)
            low, high = self.clip_bounds[attribute]
            span = high - low
            normalized = (
                np.zeros_like(values)
                if span <= 0
                else np.clip((np.clip(values, low, high) - low) / span, 0, 1)
            )
            for level in ["LOW", "MEDIUM", "HIGH"]:
                fuzzy_scores[f"{attribute}_{level}"] = _trimf(
                    normalized, self.membership_params[attribute][level]
                )

        rules = pd.DataFrame(index=result.index)
        rules["rule1_syn_ack_halfopen"] = np.minimum(
            fuzzy_scores["syn_ack_ratio_HIGH"], fuzzy_scores["half_open_conn_count_HIGH"]
        )
        rules["rule2_density_burst"] = np.minimum(
            fuzzy_scores["syn_packet_density_HIGH"], fuzzy_scores["avg_time_between_syns_LOW"]
        )
        rules["rule3_spoofed_ip"] = np.minimum(
            fuzzy_scores["same_src_ip_freq_LOW"], fuzzy_scores["syn_ack_ratio_HIGH"]
        )
        rules["rule4_retransmit_halfopen"] = np.minimum(
            fuzzy_scores["syn_packet_density_HIGH"], fuzzy_scores["half_open_conn_count_HIGH"]
        )
        rules["rule5_targeted_port"] = np.minimum(
            fuzzy_scores["unique_dst_port_count_LOW"], fuzzy_scores["syn_packet_density_HIGH"]
        )
        rules["rule6_udp_density_burst"] = np.minimum(
            fuzzy_scores["udp_packet_density_HIGH"], fuzzy_scores["total_packet_density_HIGH"]
        )
        rules["rule7_high_byte_flood"] = np.minimum(
            fuzzy_scores["byte_density_HIGH"], fuzzy_scores["total_packet_density_HIGH"]
        )
        rules["rule8_stealth_flood"] = np.minimum.reduce([
            fuzzy_scores["udp_packet_density_MEDIUM"],
            fuzzy_scores["syn_ack_ratio_MEDIUM"],
            fuzzy_scores["total_packet_density_MEDIUM"],
        ])

        rule_matrix = rules[RULE_COLUMNS].to_numpy(dtype=float)
        rules["fuzzy_confidence"] = (
            0.55 * (rule_matrix @ self.rule_weights) + 0.45 * rule_matrix.max(axis=1)
        )

        return pd.concat([result, fuzzy_scores, rules], axis=1)

    def predict(self, features: pd.DataFrame) -> dict:
        # FIX: compute enriched features ONCE, reuse for both model and fuzzy confidence
        enriched = self._fuzzy_features(features)
        model_input = enriched.reindex(columns=self.feature_order, fill_value=0).fillna(0)

        probability = float(self.model.predict_proba(model_input)[0, 1])
        fuzzy_confidence = float(enriched["fuzzy_confidence"].iloc[0])
        score = max(probability, fuzzy_confidence)

        return {
            "model_probability": probability,
            "fuzzy_confidence": fuzzy_confidence,
            "score": score,
            "threshold": self.threshold,
            "is_attack": score >= self.threshold,
        }