import torch
import torch.nn as nn
import numpy as np
import os
import glob
import joblib
from treys import Card

### --- CONFIGURATION --- ###
PROJECT_ROOT = "/home/roman/github/Poker-AI"
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")

# Must match train_limited.py exactly
INPUT_SIZE = 21
NUM_CLASSES = 3  # Fold, Call, Raise

class PokerNet(nn.Module):
    def __init__(self, input_size, num_classes):
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

    def forward(self, x):
        return self.network(x)

class PokerAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.load_latest_model()

    def load_latest_model(self):
        """Automatically finds and loads the most recent v_ OR rl_ folder"""
        if not os.path.exists(MODELS_ROOT):
            print("[AI] No models directory found.")
            return

        # 1. Find ALL version folders (Supervised v_* AND Reinforcement rl_*)
        supervised = glob.glob(os.path.join(MODELS_ROOT, "v_*"))
        reinforcement = glob.glob(os.path.join(MODELS_ROOT, "rl_*"))
        
        all_versions = supervised + reinforcement
        
        if not all_versions:
            print("[AI] No trained models found.")
            return

        # 2. Sort by creation time (newest last)
        # using os.path.getmtime to ensure the newest folder creation
        latest_dir = max(all_versions, key=os.path.getmtime)
        
        # 3. Determine filename (RL script saves as 'poker_rl_XXXX.pth', Supervised as 'poker_brain.pth')
        if "rl_" in os.path.basename(latest_dir):
            # For RL, find the latest checkpoint inside the folder
            checkpoints = glob.glob(os.path.join(latest_dir, "*.pth"))
            if not checkpoints:
                print(f"[AI] RL folder {latest_dir} is empty.")
                return
            # Sort checkpoints by number (poker_rl_500.pth, poker_rl_1000.pth)
            # Extract the number from the filename to sort correctly
            def get_ckpt_num(fname):
                try:
                    return int(fname.split('_')[-1].split('.')[0])
                except: return 0
            
            model_path = max(checkpoints, key=get_ckpt_num)
            # RL doesn't use a scaler usually (normalized inputs), but if Supervised,
            # Try to find a scaler in the PARENT 'v_' folder or skip it.
            scaler_path = None 
            # (RL typically learns to handle raw normalized inputs, so scaler might be skippable if trained enough)
            
        else:
            # Standard Supervised path
            model_path = os.path.join(latest_dir, "poker_brain.pth")
            scaler_path = os.path.join(latest_dir, "scaler.pkl")

        if not os.path.exists(model_path):
            print(f"[AI] Model file missing in {latest_dir}")
            return

        print(f"[AI] Loading Brain: {os.path.basename(latest_dir)}")
        print(f"[AI] File: {os.path.basename(model_path)}")
        
        # Load Scaler (Only if it exists)
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print("[AI] Scaler loaded.")
            except Exception as e:
                print(f"[AI] Scaler load failed: {e}")
        else:
            print("[AI] No Scaler found (Running raw inputs).")

        # Load Network
        try:
            self.model = PokerNet(INPUT_SIZE, NUM_CLASSES).to(self.device)
            # map_location ensures loading CUDA models on CPU if needed
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() 
            print("[AI] Neural Network Online.")
        except Exception as e:
            print(f"[AI] Failed to load weights: {e}")
            self.model = None

    def get_rank_suit(self, card_int):
        if card_int is None: return 0, 0
        return Card.get_rank_int(card_int), Card.get_suit_int(card_int)

    def decide(self, game_state):
        if not self.model:
            return "CALL" 

        # 1. EXTRACT FEATURES
        hole = game_state.get('hole_cards', [])
        board = game_state.get('board_cards', [])
        
        h_r1, h_s1 = self.get_rank_suit(hole[0] if len(hole) > 0 else None)
        h_r2, h_s2 = self.get_rank_suit(hole[1] if len(hole) > 1 else None)
        
        b_feats = []
        for i in range(5):
            if i < len(board):
                r, s = self.get_rank_suit(board[i])
            else:
                r, s = 0, 0
            b_feats.extend([r, s])

        raw_vector = [
            game_state.get('pot_odds', 0),
            game_state.get('spr', 0),
            game_state.get('position', 0.5),
            game_state.get('street', 0),
            game_state.get('current_pot', 0),
            game_state.get('to_call', 0),
            game_state.get('has_pair', 0),
            h_r1, h_s1, h_r2, h_s2,
            *b_feats 
        ]

        # 2. SCALE (Only if scaler exists)
        try:
            vector_np = np.array([raw_vector], dtype=np.float32)
            if self.scaler:
                vector_input = self.scaler.transform(vector_np)
            else:
                vector_input = vector_np # Pass raw if no scaler (RL often handles this)
        except Exception as e:
            print(f"[AI Error] Pre-processing failed: {e}")
            return "FOLD"

        # 3. PREDICT
        with torch.no_grad():
            input_tensor = torch.tensor(vector_input).to(self.device)
            output = self.model(input_tensor)
            action_idx = torch.argmax(output, dim=1).item()
            
            actions = ["FOLD", "CALL", "RAISE"]
            return actions[action_idx]