import torch
import torch.nn as nn
import os
import glob
import sys
import numpy as np
import joblib
from treys import Card

# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
CHAMPION_FILE = "league_champion.pth"

# Re-define Network to match training
class PokerNet(nn.Module):
    def __init__(self, input_size=21, num_classes=3):
        super(PokerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.network(x)

class PokerAI:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = PokerNet()
        self.scaler = None
        self.load_best_model()

    def load_best_model(self):
        # 1. Finding the best model
        target_path = None
        scaler_path = None
        
        champion_path = os.path.join(MODELS_DIR, CHAMPION_FILE)
        
        # Priority 1: League Champion
        if os.path.exists(champion_path):
            print(f"[AI] LOADING LEAGUE CHAMPION: {CHAMPION_FILE}")
            target_path = champion_path
            # Look for a scaler in the same dir or root
            if os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")):
                scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        else:
            # Priority 2: Newest Version
            search_pattern = os.path.join(MODELS_DIR, "v_*")
            versions = glob.glob(search_pattern)
            
            if not versions:
                print("[AI] No models found. AI is untraines.")
                return

            latest_version = max(versions, key=os.path.getctime)
            target_path = os.path.join(latest_version, "poker_brain.pth")
            scaler_path = os.path.join(latest_version, "scaler.pkl")
            print(f"[AI] Loading Newest Brain: {os.path.basename(latest_version)}")

        # 2. Load Weights
        try:
            # Handle full model vs state_dict
            try:
                self.model.load_state_dict(torch.load(target_path, map_location=self.device))
            except:
                self.model = torch.load(target_path, map_location=self.device)
            
            self.model.eval()
            print("[AI] Brain Active.")
        except Exception as e:
            print(f"[AI] Error loading brain: {e}")

        # 3. Load Scaler
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print("[AI] Vision Corrected (Scaler Loaded).")
            except Exception as e:
                print(f"[AI] Error loading scaler: {e}")
        else:
            print("[AI] WARNING: No Scaler found. AI will be blind to money amounts.")

    def decide(self, game_state):
        """
        Converts raw game state into the exact vector format the brain expects.
        """
        try:
            # 1. PARSE CARDS
            hole = game_state.get('hole_cards', [])
            board = game_state.get('board_cards', [])
            
            def get_rs(c):
                if c is None: return 0, 0
                return Card.get_rank_int(c), Card.get_suit_int(c)

            h_r1, h_s1 = get_rs(hole[0] if len(hole)>0 else None)
            h_r2, h_s2 = get_rs(hole[1] if len(hole)>1 else None)
            
            b_feats = []
            for i in range(5):
                c = board[i] if i < len(board) else None
                r, s = get_rs(c)
                b_feats.extend([r, s])

            # 2. CONSTRUCT RAW VECTOR (21 Features)
            # [pot_odds, spr, position, street, current_pot, to_call, has_pair, ...]
            
            # Auto-calculate derivatives if missing
            pot = game_state.get('pot', 0) # Raw chips
            to_call = game_state.get('to_call', 0) # Raw chips
            stack = game_state.get('stack', 1000)
            
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
            spr = stack / pot if pot > 0 else 0
            
            raw_vec = [
                pot_odds,
                spr,
                game_state.get('position', 0.5),
                game_state.get('street', 0),
                pot,      # Raw
                to_call,  # Raw
                0,        # Has Pair (Simplification: AI learns this from rank inputs)
                h_r1, h_s1, h_r2, h_s2,
                *b_feats
            ]
            
            # 3. APPLY HYBRID SCALING (Must match train_limited.py)
            vec = np.array(raw_vec, dtype=np.float32)
            
            # A. Scale Money (Indices 0, 1, 4, 5) using the loaded Scaler
            if self.scaler:
                money_indices = [0, 1, 4, 5]
                money_values = vec[money_indices].reshape(1, -1)
                scaled_money = self.scaler.transform(money_values).flatten()
                
                vec[0] = scaled_money[0]
                vec[1] = scaled_money[1]
                vec[4] = scaled_money[2]
                vec[5] = scaled_money[3]
            
            # B. Scale Cards (Divide by 13 or 4)
            rank_indices = [7, 9, 11, 13, 15, 17, 19]
            vec[rank_indices] /= 13.0
            
            suit_indices = [8, 10, 12, 14, 16, 18, 20]
            vec[suit_indices] /= 4.0

            # 4. PREDICT
            # Convert list-of-arrays to single-numpy-array first (FASTER)
            tensor_in = torch.tensor(np.array([vec]), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                output = self.model(tensor_in)
                action_idx = torch.argmax(output).item()
                
            actions = ["FOLD", "CALL", "RAISE"]
            return actions[action_idx]

        except Exception as e:
            print(f"[CRITICAL AI FAILURE] {e}")
            import traceback
            traceback.print_exc()  # Prints the line number of the error
            
            return "CALL" # Safe fallback to keep the game running