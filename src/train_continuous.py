import subprocess
import time
import os
import sys

### --- CONFIGURATION --- ###
# Use sys.executable to ensure the same Python venv running this script
PYTHON_EXE = sys.executable 
PROJECT_ROOT = "/home/roman/github/Poker-AI"
SCRIPT_TRAIN = os.path.join(PROJECT_ROOT, "src", "train_self_play.py")
SCRIPT_LEAGUE = os.path.join(PROJECT_ROOT, "src", "train_models.py")

def run_continuous_evolution():
    print("[INFO] STARTING CONTINUOUS EVOLUTION LOOP")
    print("-" * 50)
    
    loop_count = 1
    
    while True:
        print(f"\n[LOOP {loop_count}] [TRAINING] Entering the Gym...")
        print(f"[EXEC] Running: {SCRIPT_TRAIN}")
        
        try:
            # Run Training Session
            # This relies on train_self_play.py finishing after its set episodes
            subprocess.run([PYTHON_EXE, SCRIPT_TRAIN], check=True)
        except subprocess.CalledProcessError:
            print("[ERR] Training script crashed! Stopping loop.")
            break
        except KeyboardInterrupt:
            print("\n[STOP] Stopped by user.")
            break

        print(f"\n[LOOP {loop_count}] [LEAGUE] Entering the Arena...")
        print(f"[EXEC] Running: {SCRIPT_LEAGUE}")
        
        try:
            # Run League to find the best model and promote it
            subprocess.run([PYTHON_EXE, SCRIPT_LEAGUE], check=True)
        except subprocess.CalledProcessError:
            print("[ERR] League script crashed! Stopping loop.")
            break
        except KeyboardInterrupt:
            print("\n[STOP] Stopped by user.")
            break
            
        print(f"[INFO] Loop {loop_count} Complete. Starting next cycle in 5 seconds...")
        loop_count += 1
        time.sleep(5)

if __name__ == "__main__":
    run_continuous_evolution()