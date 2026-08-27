import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LabelEvent:
    epoch: float
    label: int
    name: str


@dataclass
class Interval:
    start_epoch: float
    end_epoch: float
    label: int
    name: str


def read_label_events(path: Path) -> List[LabelEvent]:
    events: List[LabelEvent] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if len(row) < 3:
                continue
            try:
                epoch = float(row[0].strip())
                label = int(row[1].strip())
                name = row[2].strip()
            except ValueError as exc:
                raise ValueError(f"Invalid labels_map row {i}: {row}") from exc
            events.append(LabelEvent(epoch=epoch, label=label, name=name))

    if not events:
        raise ValueError(f"No valid events found in {path}")

    events.sort(key=lambda e: e.epoch)
    return events


def detect_complete_cycles(events: List[LabelEvent]) -> List[Tuple[int, int, int]]:
    cycles: List[Tuple[int, int, int]] = []
    for i in range(len(events) - 2):
        triple = events[i : i + 3]
        labels = [e.label for e in triple]
        if labels == [1, 0, 1]:
            cycles.append((i, i + 1, i + 2))
    return cycles


def infer_default_duration(name: str, fallback_seconds: float) -> float:
    lower = name.lower()
    if "silence" in lower or "normal" in lower:
        return 10.0
    if "syn" in lower:
        return 20.0
    if "udp" in lower:
        return 20.0
    return fallback_seconds


def events_to_intervals(
    events: List[LabelEvent],
    start_idx: int,
    end_idx: int,
    fallback_duration: float,
) -> List[Interval]:
    intervals: List[Interval] = []
    selected = events[start_idx : end_idx + 1]

    for i, event in enumerate(selected):
        if i < len(selected) - 1:
            end_epoch = selected[i + 1].epoch
        else:
            end_epoch = event.epoch + infer_default_duration(event.name, fallback_duration)
        if end_epoch <= event.epoch:
            raise ValueError(
                f"Non-positive interval for event '{event.name}' at epoch {event.epoch}"
            )
        intervals.append(
            Interval(
                start_epoch=event.epoch,
                end_epoch=end_epoch,
                label=event.label,
                name=event.name,
            )
        )
    return intervals


def get_capture_start_epoch_from_pcap(pcap_path: Path) -> float:
    cmd = [
        "tshark",
        "-r",
        str(pcap_path),
        "-T",
        "fields",
        "-e",
        "frame.time_epoch",
        "-c",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to read capture start epoch with tshark. "
            f"stderr: {proc.stderr.strip()}"
        )

    line = proc.stdout.strip().splitlines()
    if not line:
        raise RuntimeError("tshark returned no frame.time_epoch output")

    try:
        return float(line[0].strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse frame.time_epoch: {line[0]!r}") from exc


def build_relative_intervals(
    events: List[LabelEvent],
    start_idx: int,
    end_idx: int,
    fallback_duration: float,
) -> List[Tuple[float, float, int, str]]:
    abs_intervals = events_to_intervals(
        events,
        start_idx=start_idx,
        end_idx=end_idx,
        fallback_duration=fallback_duration,
    )
    anchor = abs_intervals[0].start_epoch
    rel: List[Tuple[float, float, int, str]] = []
    for iv in abs_intervals:
        rel.append(
            (
                iv.start_epoch - anchor,
                iv.end_epoch - anchor,
                iv.label,
                iv.name,
            )
        )
    return rel


def shift_intervals(
    rel_intervals: List[Tuple[float, float, int, str]],
    offset: float,
) -> List[Tuple[float, float, int, str]]:
    return [(s + offset, e + offset, lbl, name) for s, e, lbl, name in rel_intervals]


def interval_rate_score(
    times: np.ndarray,
    intervals_rel: List[Tuple[float, float, int, str]],
) -> Tuple[float, List[int], List[float]]:
    counts: List[int] = []
    rates: List[float] = []
    labels: List[int] = []

    for start, end, label, _ in intervals_rel:
        duration = max(end - start, 1e-9)
        c = int(np.sum((times >= start) & (times < end)))
        r = c / duration
        counts.append(c)
        rates.append(r)
        labels.append(label)

    atk_rates = [r for r, l in zip(rates, labels) if l == 1]
    norm_rates = [r for r, l in zip(rates, labels) if l == 0]

    mean_atk = float(np.mean(atk_rates)) if atk_rates else 0.0
    mean_norm = float(np.mean(norm_rates)) if norm_rates else 0.0
    score = mean_atk - mean_norm
    return score, counts, rates


def find_best_cycle_and_offset(
    times: np.ndarray,
    events: List[LabelEvent],
    cycles: List[Tuple[int, int, int]],
    fallback_duration: float,
    step: float = 0.1,
) -> Tuple[int, float, List[Tuple[float, float, int, str]], dict]:
    t_min = float(np.min(times))
    t_max = float(np.max(times))

    best = None
    diagnostics = {}

    for cycle_idx, cycle in enumerate(cycles):
        start_idx, _, end_idx = cycle
        rel_tpl = build_relative_intervals(events, start_idx, end_idx, fallback_duration)
        total_span = rel_tpl[-1][1] - rel_tpl[0][0]

        # Search offsets that place most of the timeline inside observed packet times.
        offset_start = t_min - total_span
        offset_end = t_max
        if offset_end <= offset_start:
            offset_end = offset_start + step

        local_best = None
        x = offset_start
        while x <= offset_end + 1e-9:
            shifted = shift_intervals(rel_tpl, x)
            score, counts, rates = interval_rate_score(times, shifted)

            # Prefer alignments where each interval covers at least some rows.
            empty_penalty = sum(1 for c in counts if c == 0) * 1000.0
            objective = score - empty_penalty

            candidate = {
                "cycle_idx": cycle_idx,
                "offset": x,
                "objective": objective,
                "score": score,
                "counts": counts,
                "rates": rates,
                "intervals": shifted,
            }
            if local_best is None or candidate["objective"] > local_best["objective"]:
                local_best = candidate
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate
            x += step

        diagnostics[cycle_idx] = {
            "best_objective": local_best["objective"],
            "best_score": local_best["score"],
            "best_offset": local_best["offset"],
            "best_counts": local_best["counts"],
            "best_rates": [round(r, 4) for r in local_best["rates"]],
        }

    return (
        int(best["cycle_idx"]),
        float(best["offset"]),
        best["intervals"],
        diagnostics,
    )


def attach_labels(
    df: pd.DataFrame,
    intervals_rel: List[Tuple[float, float, int, str]],
) -> pd.DataFrame:
    if "Time" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column")

    out = df.copy()
    out["attack"] = 0
    out["label_scenario"] = "outside_logged_window"

    t = pd.to_numeric(out["Time"], errors="coerce")
    if t.isna().any():
        bad = int(t.isna().sum())
        raise ValueError(f"Found {bad} rows with non-numeric Time values")

    for start_rel, end_rel, label, name in intervals_rel:
        mask = (t >= start_rel) & (t < end_rel)
        if label == 1:
            out.loc[mask, "attack"] = 1
        else:
            out.loc[mask, "attack"] = 0
        out.loc[mask, "label_scenario"] = name

    return out


def summarize(df: pd.DataFrame, intervals_rel: List[Tuple[float, float, int, str]]) -> dict:
    t = pd.to_numeric(df["Time"], errors="coerce")
    min_t = float(t.min())
    max_t = float(t.max())

    rows_per_interval = []
    for start_rel, end_rel, label, name in intervals_rel:
        mask = (t >= start_rel) & (t < end_rel)
        rows = int(mask.sum())
        rows_per_interval.append(
            {
                "scenario": name,
                "label": label,
                "start_rel": round(start_rel, 6),
                "end_rel": round(end_rel, 6),
                "duration": round(end_rel - start_rel, 6),
                "rows": rows,
            }
        )

    summary = {
        "rows_total": int(len(df)),
        "time_min": min_t,
        "time_max": max_t,
        "attack_rows": int((df["attack"] == 1).sum()),
        "normal_rows": int((df["attack"] == 0).sum()),
        "attack_ratio": float((df["attack"] == 1).mean()),
        "scenario_counts": df["label_scenario"].value_counts().to_dict(),
        "rows_per_interval": rows_per_interval,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label packet CSV rows using timeline events from labels_map.txt and "
            "capture start epoch from pcap/timestamp."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="syn_udp_flood_attack_data.csv",
        help="Input packet CSV path",
    )
    parser.add_argument(
        "--labels-map",
        default="labels_map.txt",
        help="labels_map.txt path written by simulating_Attack.py",
    )
    parser.add_argument(
        "--output-csv",
        default="syn_udp_flood_attack_data_labeled.csv",
        help="Output labeled CSV path",
    )
    parser.add_argument(
        "--output-report",
        default="syn_udp_flood_attack_data_label_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--pcap",
        default="network_data.pcap",
        help="PCAP path used to derive capture start epoch via tshark",
    )
    parser.add_argument(
        "--capture-start-epoch",
        type=float,
        default=None,
        help="Override capture start epoch (seconds since epoch)",
    )
    parser.add_argument(
        "--cycle-index",
        type=int,
        default=-1,
        help=(
            "Which complete [1,0,1] cycle from labels_map to use. "
            "-1 means latest complete cycle."
        ),
    )
    parser.add_argument(
        "--fallback-duration",
        type=float,
        default=20.0,
        help="Fallback duration in seconds for last interval if next event is missing",
    )
    parser.add_argument(
        "--offset-seconds",
        type=float,
        default=None,
        help="Manually set relative offset applied to selected cycle intervals",
    )
    parser.add_argument(
        "--offset-search-step",
        type=float,
        default=0.1,
        help="Offset grid step (seconds) for auto-alignment fallback",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any interval gets 0 rows or if no rows are labeled attack",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    labels_map = Path(args.labels_map)
    output_csv = Path(args.output_csv)
    output_report = Path(args.output_report)
    pcap = Path(args.pcap)

    events = read_label_events(labels_map)
    cycles = detect_complete_cycles(events)
    if not cycles:
        raise RuntimeError(
            "No complete [1,0,1] cycles found in labels_map. "
            "Ensure simulating_Attack.py wrote all three stage start times."
        )

    cycle_idx = args.cycle_index
    if cycle_idx < 0:
        cycle_idx = len(cycles) + cycle_idx
    if cycle_idx < 0 or cycle_idx >= len(cycles):
        raise IndexError(
            f"cycle-index={args.cycle_index} out of range. "
            f"Found {len(cycles)} complete cycles."
        )

    df = pd.read_csv(input_csv)
    times = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(times).any():
        raise ValueError("Input CSV contains non-numeric Time values")

    alignment_mode = None
    alignment_meta = {}

    if args.capture_start_epoch is not None:
        capture_start_epoch = args.capture_start_epoch
        cycle = cycles[cycle_idx]
        start_idx, _, end_idx = cycle
        intervals = events_to_intervals(
            events,
            start_idx=start_idx,
            end_idx=end_idx,
            fallback_duration=args.fallback_duration,
        )
        intervals_rel = [
            (
                iv.start_epoch - capture_start_epoch,
                iv.end_epoch - capture_start_epoch,
                iv.label,
                iv.name,
            )
            for iv in intervals
        ]
        alignment_mode = "manual_capture_start_epoch"
        alignment_meta = {"capture_start_epoch": capture_start_epoch, "cycle_index": cycle_idx}
    else:
        pcap_ok = False
        capture_start_epoch = None
        try:
            capture_start_epoch = get_capture_start_epoch_from_pcap(pcap)
            pcap_ok = True
        except Exception:
            pcap_ok = False

        if pcap_ok:
            cycle = cycles[cycle_idx]
            start_idx, _, end_idx = cycle
            intervals = events_to_intervals(
                events,
                start_idx=start_idx,
                end_idx=end_idx,
                fallback_duration=args.fallback_duration,
            )
            intervals_rel = [
                (
                    iv.start_epoch - capture_start_epoch,
                    iv.end_epoch - capture_start_epoch,
                    iv.label,
                    iv.name,
                )
                for iv in intervals
            ]
            alignment_mode = "pcap_capture_start_epoch"
            alignment_meta = {
                "capture_start_epoch": capture_start_epoch,
                "cycle_index": cycle_idx,
                "pcap": str(pcap),
            }
        else:
            # Fallback: infer relative offset from packet-rate contrast.
            if args.cycle_index >= 0:
                cycle = cycles[cycle_idx]
                start_idx, _, end_idx = cycle
                rel_tpl = build_relative_intervals(
                    events,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    fallback_duration=args.fallback_duration,
                )

                if args.offset_seconds is None:
                    auto_cycle_idx, auto_offset, auto_intervals, diagnostics = find_best_cycle_and_offset(
                        times,
                        events,
                        [cycle],
                        args.fallback_duration,
                        step=args.offset_search_step,
                    )
                    intervals_rel = auto_intervals
                    alignment_mode = "auto_offset_single_cycle"
                    alignment_meta = {
                        "selected_cycle_index": cycle_idx,
                        "best_offset_seconds": auto_offset,
                        "offset_search_step": args.offset_search_step,
                        "diagnostics": diagnostics,
                    }
                else:
                    intervals_rel = shift_intervals(rel_tpl, args.offset_seconds)
                    alignment_mode = "manual_offset_single_cycle"
                    alignment_meta = {
                        "selected_cycle_index": cycle_idx,
                        "manual_offset_seconds": args.offset_seconds,
                    }
            else:
                auto_cycle_idx, auto_offset, auto_intervals, diagnostics = find_best_cycle_and_offset(
                    times,
                    events,
                    cycles,
                    args.fallback_duration,
                    step=args.offset_search_step,
                )
                intervals_rel = auto_intervals
                alignment_mode = "auto_cycle_auto_offset"
                alignment_meta = {
                    "selected_cycle_index": auto_cycle_idx,
                    "best_offset_seconds": auto_offset,
                    "offset_search_step": args.offset_search_step,
                    "diagnostics": diagnostics,
                }

    labeled = attach_labels(df, intervals_rel)
    report = summarize(labeled, intervals_rel)
    report["alignment_mode"] = alignment_mode
    report["alignment_meta"] = alignment_meta

    interval_rows = [item["rows"] for item in report["rows_per_interval"]]
    if args.strict:
        if any(r == 0 for r in interval_rows):
            raise RuntimeError(
                "Strict mode failed: at least one timeline interval matched 0 rows. "
                "Check capture start epoch / cycle selection."
            )
        if report["attack_rows"] == 0:
            raise RuntimeError("Strict mode failed: no attack rows were labeled.")

    labeled.to_csv(output_csv, index=False)
    with output_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Labeling complete.")
    print(f"Input rows: {report['rows_total']}")
    print(f"Attack rows: {report['attack_rows']} | Normal rows: {report['normal_rows']}")
    print(f"Attack ratio: {report['attack_ratio']:.4f}")
    print(f"Alignment mode: {alignment_mode}")
    if "selected_cycle_index" in alignment_meta:
        print(
            f"Used complete cycle index: {alignment_meta['selected_cycle_index']} "
            f"(from {len(cycles)} detected cycles)"
        )
    elif "cycle_index" in alignment_meta:
        print(
            f"Used complete cycle index: {alignment_meta['cycle_index']} "
            f"(from {len(cycles)} detected cycles)"
        )
    if "best_offset_seconds" in alignment_meta:
        print(f"Estimated offset (seconds): {alignment_meta['best_offset_seconds']:.3f}")
    if "manual_offset_seconds" in alignment_meta:
        print(f"Manual offset (seconds): {alignment_meta['manual_offset_seconds']:.3f}")
    print("Rows per scenario interval:")
    for row in report["rows_per_interval"]:
        print(
            f"  - {row['scenario']} (label={row['label']}): "
            f"[{row['start_rel']:.3f}, {row['end_rel']:.3f}) sec -> {row['rows']} rows"
        )
    print(f"Wrote labeled CSV: {output_csv}")
    print(f"Wrote report JSON: {output_report}")


if __name__ == "__main__":
    main()
