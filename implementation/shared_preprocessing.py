import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _extract_ports(info_value: str) -> tuple[float, float]:
    match = re.search(r"(\d+)\s+>\s+(\d+)", str(info_value))
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan


def _rolling_src_freq(source_series: pd.Series, window: int) -> list[float]:
    result: list[float] = []
    for i in range(len(source_series)):
        start = max(0, i - window + 1)
        window_slice = source_series.iloc[start : i + 1]
        current_src = source_series.iloc[i]
        result.append(float((window_slice == current_src).sum() / len(window_slice)))
    return result


def _rolling_unique_dst_ports(dst_port_series: pd.Series, window: int) -> list[int]:
    result: list[int] = []
    for i in range(len(dst_port_series)):
        start = max(0, i - window + 1)
        window_slice = dst_port_series.iloc[start : i + 1].dropna()
        result.append(int(window_slice.nunique()))
    return result


def _avg_syn_interarrival(df_input: pd.DataFrame, window: int) -> list[float]:
    result: list[float] = []
    syn_times: list[float] = []
    for i in range(len(df_input)):
        if int(df_input["syn_flag"].iloc[i]) == 1:
            syn_times.append(float(df_input["Time"].iloc[i]))

        start_idx = max(0, i - window + 1)
        start_time = float(df_input["Time"].iloc[start_idx])
        syn_times = [t for t in syn_times if t >= start_time]

        if len(syn_times) >= 2:
            result.append(float(np.mean(np.diff(syn_times))))
        else:
            result.append(10.0)
    return result


def build_clean_dataset(
    input_csv: str = "data/syn_udp_flood_attack_data_labeled.csv",
    output_csv: str = "data/syn_udp_flood_attack_data_clean.csv",
    output_report: str = "reports/clean_dataset_report.json",
    window: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    input_path = Path(input_csv)
    if not input_path.exists():
        input_path = Path("syn_udp_flood_attack_data_labeled.csv")

    df = pd.read_csv(input_path)

    for col in ["Time", "Length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "attack" not in df.columns:
        raise ValueError("Missing required 'attack' column in labeled dataset")

    if not pd.api.types.is_numeric_dtype(df["attack"]):
        mapped = df["attack"].astype(str).str.lower().map(
            {
                "attack": 1, "malicious": 1, "anomaly": 1, "1": 1, "true": 1,
                "normal": 0, "benign": 0, "0": 0, "false": 0,
            }
        )
        df["attack"] = mapped.fillna(0).astype(int)
    else:
        df["attack"] = (pd.to_numeric(df["attack"], errors="coerce").fillna(0) > 0).astype(int)

    df = df.dropna(subset=["Time", "Length", "Info", "Protocol", "Source", "Destination"]).copy()
    df = df.sort_values("Time").reset_index(drop=True)

    info = df["Info"].astype(str)
    df["syn_flag"] = info.str.contains(r"\[SYN\]", regex=True, na=False).astype(int)
    df["ack_flag"] = info.str.contains(r"\[ACK\]", regex=True, na=False).astype(int)
    df["syn_ack_flag"] = info.str.contains(r"\[SYN, ACK\]", regex=True, na=False).astype(int)
    df["is_retransmission"] = info.str.contains("Retransmission", na=False).astype(int)

    df["is_tcp"] = (df["Protocol"].astype(str) == "TCP").astype(int)
    df["is_udp"] = (df["Protocol"].astype(str) == "UDP").astype(int)

    ports = df["Info"].apply(_extract_ports)
    df["src_port"] = ports.apply(lambda x: x[0])
    df["dst_port"] = ports.apply(lambda x: x[1])

    df["packet_length"] = df["Length"].astype(float)
    df["time_delta"] = df["Time"].diff().fillna(0).clip(lower=0)
    df["is_syn_only"] = ((df["syn_flag"] == 1) & (df["syn_ack_flag"] == 0)).astype(int)

    syn_window = df["syn_flag"].rolling(window=window, min_periods=1).sum()
    ack_window = df["ack_flag"].rolling(window=window, min_periods=1).sum()
    udp_window = df["is_udp"].rolling(window=window, min_periods=1).sum()
    bytes_window = df["packet_length"].rolling(window=window, min_periods=1).sum()

    window_time = df["Time"].rolling(window=window, min_periods=1).apply(
        lambda x: max(float(x.iloc[-1] - x.iloc[0]), 0.0001), raw=False
    )

    df["syn_ack_ratio"] = syn_window / (ack_window + 1)
    df["half_open_conn_count"] = df["is_syn_only"].rolling(window=window, min_periods=1).sum()
    df["same_src_ip_freq"] = _rolling_src_freq(df["Source"], window)
    df["syn_packet_density"] = (syn_window / window_time).fillna(0).clip(upper=1000)
    df["udp_packet_density"] = (udp_window / window_time).fillna(0).clip(upper=1000)
    df["total_packet_density"] = (window / window_time).fillna(0).clip(upper=2000)
    df["byte_density"] = (bytes_window / window_time).fillna(0).clip(upper=1000000)
    df["retransmission_rate"] = df["is_retransmission"].rolling(window=window, min_periods=1).mean()
    df["unique_dst_port_count"] = _rolling_unique_dst_ports(df["dst_port"], window)
    df["avg_time_between_syns"] = _avg_syn_interarrival(df, window)
    df["pkt_len_mean"] = df["packet_length"].rolling(window=window, min_periods=1).mean()
    df["pkt_len_std"] = df["packet_length"].rolling(window=window, min_periods=1).std().fillna(0)
    df["udp_ratio"] = udp_window / window
    df["syn_ratio"] = syn_window / window

    df["label_scenario"] = df.get("label_scenario", "unknown").astype(str)

    strat_key = np.where(
        df["attack"] == 1,
        df["label_scenario"],
        "normal_background",
    )
    train_idx, test_idx = train_test_split(
        df.index.values,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )

    df["split"] = "train"
    df.loc[np.sort(test_idx), "split"] = "test"

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "rows_total": int(len(df)),
        "attack_rows": int((df["attack"] == 1).sum()),
        "normal_rows": int((df["attack"] == 0).sum()),
        "attack_ratio": float(df["attack"].mean()),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_attack_ratio": df.groupby("split")["attack"].mean().to_dict(),
        "window": int(window),
        "random_state": int(random_state),
    }

    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean shared dataset with enriched SYN & UDP flood features")
    parser.add_argument("--input-csv", default="data/syn_udp_flood_attack_data_labeled.csv")
    parser.add_argument("--output-csv", default="data/syn_udp_flood_attack_data_clean.csv")
    parser.add_argument("--output-report", default="reports/clean_dataset_report.json")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    _, report = build_clean_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_report=args.output_report,
        window=args.window,
    )

    print("Clean dataset created with enriched SYN & UDP flood features.")
    print(f"Rows: {report['rows_total']} | Output: {report['output_csv']}")


if __name__ == "__main__":
    main()
