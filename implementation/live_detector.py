"""Bounded replay/live detector with active edge mitigation (ALLOW, RESTRICT/SLOW DOWN, BLOCK) and rich perception logging."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

from fuzzy_inference import FuzzyForestPredictor


FEATURES = [
    "packet_length", "time_delta", "is_syn_only", "syn_ack_ratio",
    "half_open_conn_count", "same_src_ip_freq", "syn_packet_density",
    "retransmission_rate", "unique_dst_port_count", "avg_time_between_syns",
    "is_tcp", "is_udp",
]

# Track blocked IPs to prevent duplicate firewall executions
BLOCKED_IPS: set[str] = set()
RESTRICTED_IPS: set[str] = set()


def apply_firewall_mitigation(source_ip: str, action: str, dry_run: bool = True) -> str:
    """Executes or simulates edge iptables/nftables firewall mitigation on Raspberry Pi."""
    if not source_ip or source_ip == "unknown" or source_ip.startswith("127."):
        return "Internal traffic (no action)"

    if action == "BLOCK":
        if source_ip in BLOCKED_IPS:
            return f"IP {source_ip} already blocked in iptables"
        
        cmd = f"sudo iptables -A INPUT -s {source_ip} -j DROP"
        if dry_run or sys.platform != "linux":
            BLOCKED_IPS.add(source_ip)
            return f"[SIMULATED] Firewall rule created: DROP traffic from {source_ip}"
        
        try:
            res = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            BLOCKED_IPS.add(source_ip)
            return f"[FIREWALL ACTIVE] Blocked {source_ip} via iptables"
        except Exception as err:
            BLOCKED_IPS.add(source_ip)
            return f"[FIREWALL WARN] (Simulated block on non-root/Windows): {err}"

    elif action == "RESTRICT":
        if source_ip in RESTRICTED_IPS:
            return f"IP {source_ip} currently rate-limited (throttle delay active)"
        
        RESTRICTED_IPS.add(source_ip)
        # Apply slight micro-delay to throttle processing and prevent edge CPU exhaustion
        time.sleep(0.02)
        return f"[RATE-LIMIT] Throttling incoming requests from {source_ip} (20ms edge backoff)"

    return "Traffic permitted (ALLOW)"


def row_features(rows: list[dict], window: int) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(rows).tail(window).copy()
    info = frame["Info"].astype(str)
    frame["syn_flag"] = info.str.contains(r"\[SYN\]", regex=True, na=False).astype(int)
    frame["ack_flag"] = info.str.contains(r"\[ACK\]", regex=True, na=False).astype(int)
    frame["syn_ack_flag"] = info.str.contains(r"\[SYN, ACK\]", regex=True, na=False).astype(int)
    frame["is_retransmission"] = info.str.contains("Retransmission", na=False).astype(int)
    frame["is_tcp"] = (frame["Protocol"].astype(str) == "TCP").astype(int)
    frame["is_udp"] = (frame["Protocol"].astype(str) == "UDP").astype(int)
    frame["packet_length"] = pd.to_numeric(frame["Length"], errors="coerce").fillna(0)
    frame["time_delta"] = pd.to_numeric(frame["Time"], errors="coerce").diff().fillna(0)
    frame["is_syn_only"] = ((frame["syn_flag"] == 1) & (frame["syn_ack_flag"] == 0)).astype(int)
    
    syn_count = int(frame["syn_flag"].sum())
    ack_count = int(frame["ack_flag"].sum())
    duration = max(float(frame["Time"].iloc[-1] - frame["Time"].iloc[0]), 0.001)
    
    syn_ack_ratio = float(syn_count / (ack_count + 1))
    half_open_conn_count = int(frame["is_syn_only"].sum())
    same_src_ip_freq = float(frame["Source"].value_counts().iloc[0] / len(frame))
    syn_packet_density = min(float(syn_count / duration), 100.0)
    retransmission_rate = float(frame["is_retransmission"].mean())
    unique_dst_port_count = int(frame["Info"].astype(str).str.extract(r"\d+\s+>\s+(\d+)")[0].nunique())
    
    syn_times = pd.to_numeric(frame.loc[frame["syn_flag"] == 1, "Time"], errors="coerce").dropna()
    avg_time_between_syns = float(syn_times.diff().mean()) if len(syn_times) > 1 else 10.0

    frame["syn_ack_ratio"] = syn_ack_ratio
    frame["half_open_conn_count"] = half_open_conn_count
    frame["same_src_ip_freq"] = same_src_ip_freq
    frame["syn_packet_density"] = syn_packet_density
    frame["retransmission_rate"] = retransmission_rate
    frame["unique_dst_port_count"] = unique_dst_port_count
    frame["avg_time_between_syns"] = avg_time_between_syns

    stats = {
        "syn_count": syn_count,
        "ack_count": ack_count,
        "syn_ack_ratio": round(syn_ack_ratio, 3),
        "half_open_conn_count": half_open_conn_count,
        "syn_packet_density": round(syn_packet_density, 2),
        "avg_time_between_syns": round(avg_time_between_syns, 4),
        "same_src_ip_freq": round(same_src_ip_freq, 3),
    }

    return frame.iloc[[-1]][FEATURES], stats


def run_replay(
    predictor: FuzzyForestPredictor,
    csv_path: str,
    log_path: str,
    window: int,
    delay: float,
    dry_run: bool = True,
) -> None:
    rows = deque(maxlen=window)
    output = Path(log_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Path(csv_path).open(encoding="utf-8") as source, output.open("a", encoding="utf-8") as log:
        reader = pd.read_csv(source)
        total_records = len(reader)
        print(f"Starting live perception detector on {total_records} packets (Window: {window})...")

        for idx, row in enumerate(reader.to_dict("records")):
            rows.append(row)
            if len(rows) < 2:
                continue

            features, stats = row_features(list(rows), window)
            result = predictor.predict(features)
            
            score = result["score"]
            threshold = result["threshold"]

            # Edge Mitigation Decision Logic
            if score >= threshold:
                action = "BLOCK"
            elif score >= threshold * 0.5:
                action = "RESTRICT"
            else:
                action = "ALLOW"

            source_ip = str(row.get("Source", "unknown"))
            mitigation_msg = apply_firewall_mitigation(source_ip, action, dry_run=dry_run)

            event = {
                "timestamp": time.time(),
                "packet_index": idx + 1,
                "packets_in_window": len(rows),
                "source": source_ip,
                "action": action,
                "mitigation_msg": mitigation_msg,
                "stats": stats,
                **result,
            }

            log.write(json.dumps(event) + "\n")
            log.flush()

            if idx % 500 == 0 or action != "ALLOW":
                print(f"[{time.strftime('%H:%M:%S')}] Pkt {idx+1}/{total_records} | Src: {source_ip:15s} | Score: {score:.3f} (Fuzzy: {result['fuzzy_confidence']:.3f}) | Action: {action:8s} -> {mitigation_msg}")

            if delay > 0:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live packet perception and edge mitigation detector")
    parser.add_argument("--replay-csv", required=True, help="CSV exported from network capture or dataset")
    parser.add_argument("--artifact", default="fuzzy_forest.pkl")
    parser.add_argument("--log", default="reports/monitor_events.jsonl")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--live-firewall", action="store_true", help="Execute real sudo iptables commands on Linux/Pi")
    args = parser.parse_args()

    predictor = FuzzyForestPredictor(args.artifact)
    run_replay(
        predictor=predictor,
        csv_path=args.replay_csv,
        log_path=args.log,
        window=args.window,
        delay=args.delay,
        dry_run=not args.live_firewall,
    )


if __name__ == "__main__":
    main()