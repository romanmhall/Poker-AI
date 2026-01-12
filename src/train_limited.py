import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os
import numpy as np
import joblib
import datetime
import gc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

### --- CONFIGURATION (TUNED FOR 32GB RAM) --- ###
PROJECT_ROOT = "/home/roman/github/Poker-AI"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "training_ready")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")

# 500 files ~= 8.5 GB RAM. 
# If system lags, lower to 300. If plenty of RAM, try 800.
FILE_BUFFER_SIZE = 500  
BATCH_SIZE = 4096
EPOCHS = 5             
LEARNING_RATE = 0.001
INPUT_SIZE = 21

# Output Setup
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION_DIR = os.path.join(MODELS_ROOT, f"v_{timestamp}")
os.makedirs(VERSION_DIR, exist_ok=True)
MODEL_PATH = os.path.join(VERSION_DIR, "poker_brain.pth")
SCALER_PATH = os.path.join(VERSION_DIR, "scaler.pkl")

### --- 1. NETWORK --- ###
class PokerNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PokerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),       # Wider layer
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

### --- 2. DATA UTILS --- ###
def load_file_chunk(file_paths, scaler=None, fit_scaler=False):
    """Loads a list of Parquet files into one giant DataFrame"""
    dfs = []
    for f in file_paths:
        try:
            dfs.append(pd.read_parquet(f))
        except: pass
    
    if not dfs: return None, None
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning
    df = df.dropna(subset=['target_action'])
    action_map = {'FOLD': 0, 'fold': 0, 'call': 1, 'CALL': 1, 'check': 1, 'CHECK': 1, 'bet': 2, 'BET': 2, 'raise': 2, 'RAISE': 2}
    df['label'] = df['target_action'].map(action_map)
    df = df.dropna(subset=['label'])
    
    # Feature Extraction
    feature_cols = [
        'pot_odds', 'spr', 'position', 'street', 'current_pot', 'to_call', 'has_pair',
        'hole_rank_1', 'hole_suit_1', 'hole_rank_2', 'hole_suit_2',
        'board_rank_1', 'board_suit_1', 'board_rank_2', 'board_suit_2',
        'board_rank_3', 'board_suit_3', 'board_rank_4', 'board_suit_4',
        'board_rank_5', 'board_suit_5'
    ]
    
    # Fill missing cols with 0
    for c in feature_cols:
        if c not in df.columns: df[c] = 0

    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    
    # Scaler Logic
    if fit_scaler and scaler:
        scaler.partial_fit(X)
    
    if not fit_scaler and scaler:
        X = scaler.transform(X)
        
    return X, y

def train_buffered():
    print(f"[INFO] Starting BUFFERED Training (Using ~10GB RAM Chunks)")
    
    # 1. Discover Files
    all_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    np.random.shuffle(all_files) # Global shuffle
    print(f"[INFO] Found {len(all_files)} files.")
    
    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")
    
    model = PokerNet(input_size=INPUT_SIZE, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Calibration Pass (First 300 files to learn scaling)
    print("[INFO] Calibrating Scaler (Scanning sample data)...")
    scaler = StandardScaler()
    X_cal, _ = load_file_chunk(all_files[:300], scaler=scaler, fit_scaler=True)
    joblib.dump(scaler, SCALER_PATH)
    del X_cal
    gc.collect()
    print("[INFO] Scaler calibrated.")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        np.random.shuffle(all_files) # Re-shuffle every epoch
        
        # Process in Mega-Chunks
        total_chunks = len(all_files) // FILE_BUFFER_SIZE + 1
        
        for i in range(0, len(all_files), FILE_BUFFER_SIZE):
            chunk_files = all_files[i : i + FILE_BUFFER_SIZE]
            chunk_id = (i // FILE_BUFFER_SIZE) + 1
            
            print(f"   [Chunk {chunk_id}/{total_chunks}] Loading {len(chunk_files)} files...", end="\r")
            
            # Load ~8GB into RAM
            X_chunk, y_chunk = load_file_chunk(chunk_files, scaler=scaler)
            if X_chunk is None: continue
            
            print(f"   [Chunk {chunk_id}/{total_chunks}] Training on {len(X_chunk):,} hands...   ", end="\r")
            
            # Convert to Tensor (Moves to GPU in batches)
            tensor_x = torch.tensor(X_chunk)
            tensor_y = torch.tensor(y_chunk)
            
            dataset = TensorDataset(tensor_x, tensor_y)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffling here
            
            # Sub-Loop for this RAM chunk
            model.train()
            chunk_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                chunk_loss += loss.item()
            
            # Cleanup RAM immediately
            del X_chunk, y_chunk, tensor_x, tensor_y, dataset, loader
            gc.collect() 
            
            print(f"   [Chunk {chunk_id}/{total_chunks}] Done. Avg Loss: {chunk_loss:.4f}")
        
        # Checkpoint every epoch
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n[SUCCESS] TRAINING COMPLETE!")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_buffered()