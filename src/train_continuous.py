import os
import time
import shutil
import glob
import sys
import torch
import torch.nn as nn
import re
import csv
import datetime
import statistics
from train_self_play import SelfPlayEngine
from train_models import LeagueTournament
from train_limited import train_champion

# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
ROSTER_DIR = os.path.join(MODELS_DIR, "active_roster")
CHAMPION_PATH = os.path.join(MODELS_DIR, "league_champion.pth")
CHAMPION_NAME_FILE = os.path.join(MODELS_DIR, "champion_name.txt")
ROSTER_LOG_FILE = os.path.join(PROJECT_ROOT, "data", "roster_history.csv")

# Archive for train_grandmaster 
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "data", "training_ready", "consolidated")
if not os.path.exists(ARCHIVE_DIR): os.makedirs(ARCHIVE_DIR)

# HELPER CLASSES
class PokerNet(nn.Module):
    def __init__(self, input_size=21, num_classes=3):
        super(PokerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.2), 
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.network(x)

class RosterLogger:
    def __init__(self):
        if not os.path.exists(ROSTER_LOG_FILE):
            with open(ROSTER_LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Cycle", "Action", "Model_Name", "CPI_Score", "Reason"])

    def log(self, cycle, action, name, score, reason):
        with open(ROSTER_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cycle, action, name, score, reason
            ])

# HELPER FUNCTIONS
def create_mutant(source_path, dest_path, mutation_power=0.02):
    try:
        model = PokerNet()
        # Load CPU to ensure compatibility
        model.load_state_dict(torch.load(source_path, map_location='cpu'))
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * mutation_power
                param.add_(noise)
        torch.save(model.state_dict(), dest_path)
        return True
    except Exception as e:
        print(f"[ERR] Mutation Failed: {e}")
        return False

def get_bot_cpi(filepath):
    """Extracts CPI (0-100) from filename. Default 10 if unknown."""
    # Format: Bot_cpi85.pth
    match = re.search(r'_cpi(\d+)\.pth', filepath)
    if match: return int(match.group(1))
    
    # Old format: Bot_s3.pth (Map 0-4 to 0-100)
    match_old = re.search(r'_s(\d)\.pth', filepath)
    if match_old: return int(match_old.group(1)) * 25
    
    return 10 # Default low score for untagged bots

def genesis_protocol():
    """Checks the roster health and auto-fills it if empty."""
    if not os.path.exists(ROSTER_DIR): os.makedirs(ROSTER_DIR)
    
    # Ensure Patient Zero exists
    if not os.path.exists(CHAMPION_PATH):
        print("[GENESIS] No Champion found! Creating seed.")
        model = PokerNet()
        torch.save(model.state_dict(), CHAMPION_PATH)
        with open(CHAMPION_NAME_FILE, 'w') as f: f.write("Random_Seed_Gen0")
    
    # Check Population
    files = glob.glob(os.path.join(ROSTER_DIR, "*.pth"))
    if len(files) < 6:
        needed = 6 - len(files)
        print(f"[GENESIS] Deploying {needed} reinforcements...")
        for i in range(needed):
            # Create filler bots with low CPI tag so they are culled first
            name = f"Reinforce_{int(time.time())}_{i}_cpi10.pth"
            create_mutant(CHAMPION_PATH, os.path.join(ROSTER_DIR, name), 0.05)

def print_evolution_health():
    """Reads the log and advises on gene pool health."""
    if not os.path.exists(ROSTER_LOG_FILE): return
    scores = []
    with open(ROSTER_LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Action'] == "CULLED":
                try: scores.append(int(float(row['CPI_Score'])))
                except: pass
    
    recent = scores[-5:]
    if not recent: return
    avg = statistics.mean(recent)
    print(f"\n   [COACH] Recent Culls (Avg CPI): {avg:.1f}/100")
    if avg < 40: print("   -> HEALTHY. Weeding out weak bots.")
    elif avg > 70: print("   -> CRITICAL. Deleting Pro bots! Check Roster size.")

# MAIN LOOP
def main():
    logger = RosterLogger()
    print("\n[INFO] STARTING CPI EVOLUTION LOOP (With Archiving)")
    loop_count = 1
    stagnation_counter = 0
    last_champ = ""

    try:
        while True:
            # STEP 0: Auto-Refill Roster
            genesis_protocol()
            
            print(f"\n" + "="*40 + f"\n   CYCLE #{loop_count} | Stagnation: {stagnation_counter}\n" + "="*40)
            
            # PHASE 1: SELF PLAY
            print("\n[PHASE 1] Generating Self-Play Data...")
            engine = SelfPlayEngine()
            for i in range(10): 
                engine.play_batch()
                sys.stdout.write(f"\r   -> Batch {i+1}/10")
                sys.stdout.flush()
            
            # PHASE 2: TRAINING & ARCHIVING
            print("\n\n[PHASE 2] Training Neural Network...")
            data_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "training_ready", "self_play", "*.parquet"))
            
            if data_files:
                # 1. Train the Quick Learner (Champion)
                train_champion(CHAMPION_PATH, data_files, epochs=2)
                
                # 2. Archive for the Deep Learner (Grandmaster)
                print(f"   -> Archiving {len(data_files)} files for Grandmaster...")
                for f in data_files: 
                    try:
                        base_name = os.path.basename(f)
                        shutil.move(f, os.path.join(ARCHIVE_DIR, base_name))
                    except Exception as e:
                        print(f"[WARN] Failed to archive {base_name}: {e}")
            else:
                print("   [INFO] No new data found.")
            
            # PHASE 3: TOURNAMENT
            print("\n[PHASE 3] League Tournament...")
            tournament = LeagueTournament()
            stats = tournament.run()
            
            # PHASE 4: EVOLUTION
            print("\n[PHASE 4] Evolution...")
            mutants = 0
            severity = 0.02
            reason = "Standard"
            
            if stats:
                cpi = stats.get('cpi', 0.0)
                # THRESHOLD: 0.60 CPI (Roughly Score 2/4)
                if cpi < 0.60: 
                    stagnation_counter += 1
                    mutants = 2
                    severity = 0.05
                    reason = f"Low CPI ({cpi:.2f})"
                elif stats['winner_name'] == last_champ:
                    stagnation_counter += 1
                    if stagnation_counter >= 3:
                        mutants = 3
                        severity = 0.10
                        reason = "Stagnation"
                    else:
                        mutants = 1
                        reason = "Selection"
                else:
                    stagnation_counter = 0
                    mutants = 1
                    reason = "New King"
                last_champ = stats['winner_name']

            if mutants > 0:
                files = glob.glob(os.path.join(ROSTER_DIR, "*.pth"))
                # Sort by CPI (Lowest first) -> First to die
                files.sort(key=lambda f: (get_bot_cpi(f), os.path.getmtime(f)))
                
                print("   -> Roster Snapshot:")
                for f in files: print(f"      [CPI {get_bot_cpi(f)}] {os.path.basename(f)}")
                
                # CULLING
                while len(files) + mutants > 6:
                    if files:
                        victim = files.pop(0)
                        score = get_bot_cpi(victim)
                        try:
                            os.remove(victim)
                            logger.log(loop_count, "CULLED", os.path.basename(victim), score, reason)
                            print(f"      -> Culled: {os.path.basename(victim)}")
                        except: pass
                
                # SPAWNING
                for i in range(mutants):
                    # Spawn with CPI 0 tag so they have to prove themselves
                    name = f"Mutant_C{loop_count}_{i}_cpi0.pth"
                    create_mutant(CHAMPION_PATH, os.path.join(ROSTER_DIR, name), severity)
                    logger.log(loop_count, "SPAWNED", name, 0, reason)

            print_evolution_health()
            loop_count += 1
            time.sleep(5)

    except KeyboardInterrupt: print("\n[STOP] Loop terminated.")

if __name__ == "__main__":
    main()