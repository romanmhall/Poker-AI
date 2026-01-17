import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob  # <--- Fixed missing import

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")

# NEW PATH: analysis/visualizations
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "visualizations")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# --- NETWORK ARCHITECTURE ---
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

def get_preflop_vector(rank1, suit1, rank2, suit2):
    """Creates a standardized state vector for 'Pre-Flop, First to Act'."""
    pot_odds = 0.33  
    spr = 100.0      
    position = 0.5   
    street = 0       
    pot = 30         
    to_call = 20     
    has_pair = 1.0 if rank1 == rank2 else 0.0
    
    r1, r2 = rank1 / 13.0, rank2 / 13.0
    s1, s2 = suit1 / 4.0, suit2 / 4.0
    
    vec = [pot_odds, spr, position, len([]), pot, to_call, has_pair,
           r1, s1, r2, s2,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    
    return np.array(vec, dtype=np.float32)

def generate_heatmap(model_path, title_suffix=""):
    print(f"Analyzing Strategy for: {os.path.basename(model_path)}")
    
    model = PokerNet()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except:
        print("[ERR] Could not load model. Architecture mismatch?")
        return

    model.eval()
    
    # 13x13 Grid setup
    strategy_grid = np.zeros((13, 13))
    ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    
    for i in range(13): 
        for j in range(13): 
            if i > j: # Offsuit
                r1, s1 = i, 0 
                r2, s2 = j, 1 
            elif i < j: # Suited
                r1, s1 = i, 0 
                r2, s2 = j, 0 
            else: # Pairs
                r1, s1 = i, 0
                r2, s2 = j, 1
            
            state = get_preflop_vector(r1, s1, r2, s2)
            tensor = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                action = torch.argmax(logits).item()
            
            strategy_grid[12-i, 12-j] = action

    # PLOT
    plt.figure(figsize=(10, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#ffdddd', '#ffffdd', '#ddffdd']) # Red, Yellow, Green
    
    labels = list(reversed(ranks))
    sns.heatmap(strategy_grid, annot=True, cmap=cmap, cbar=False,
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray')
    
    plt.title(f"Starting Hand Strategy\n{os.path.basename(model_path)}")
    plt.xlabel("Suited (Upper) / Offsuit (Lower)")
    plt.ylabel("High Card")
    plt.figtext(0.5, 0.02, "0 (Red) = Fold | 1 (Yellow) = Call | 2 (Green) = Raise", ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    save_path = os.path.join(OUTPUT_DIR, f"heatmap_{title_suffix}.png")
    plt.savefig(save_path)
    print(f"[SUCCESS] Saved heatmap to {save_path}")
    plt.close()

if __name__ == "__main__":
    # 1. Visualize Current Champion
    champ_path = os.path.join(MODELS_DIR, "league_champion.pth")
    if os.path.exists(champ_path):
        generate_heatmap(champ_path, "current_champion")
    
    # 2. Visualize Grandmaster
    gm_dirs = sorted(glob.glob(os.path.join(MODELS_DIR, "v_GRANDMASTER_*")))
    if gm_dirs:
        # Check inside the latest folder for the .pth file
        latest_gm_folder = gm_dirs[-1]
        latest_gm_file = os.path.join(latest_gm_folder, "poker_brain.pth")
        
        if os.path.exists(latest_gm_file):
            generate_heatmap(latest_gm_file, "grandmaster")
        else:
            print(f"[WARN] Found folder {latest_gm_folder} but no 'poker_brain.pth' inside.")
    else:
        print("[INFO] No Grandmaster models found yet.")
            
    print(f"\n[INFO] Done. Check '{OUTPUT_DIR}' for images.")