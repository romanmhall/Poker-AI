import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os
import numpy as np
import joblib
import datetime
import time
import csv
import sys
import gc  # Garbage Collector
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "training_ready", "consolidated") 
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")
LOG_FILE = os.path.join(PROJECT_ROOT, "data", "training_history.csv")
EPOCH_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "grandmaster_epoch.csv") # <--- NEW LOCATION

# HYPERPARAMETERS - Optimized for 3070/32G RAM
BATCH_SIZE = 65536
EPOCHS = 50             
LEARNING_RATE = 0.0001 
INPUT_SIZE = 21
CHUNK_SIZE = 1000

# SETUP
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION_DIR = os.path.join(MODELS_ROOT, f"v_GRANDMASTER_{timestamp}")
os.makedirs(VERSION_DIR, exist_ok=True)
MODEL_PATH = os.path.join(VERSION_DIR, "poker_brain.pth")
SCALER_PATH = os.path.join(VERSION_DIR, "scaler.pkl")

# NETWORK
class PokerNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PokerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.2), 
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.network(x)

def log_training_run(model_name, source_count, epochs, final_loss, duration):
    """Logs the final summary of the entire run."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model_Name", "Source_Data", "Epochs", "Final_Loss", "Duration_Sec"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sources = f"{source_count} Files (Chunked)" 
        writer.writerow([timestamp, model_name, sources, epochs, f"{final_loss:.4f}", f"{duration:.1f}"])

def log_epoch(epoch, loss):
    """Logs the loss for a specific epoch (for graphing curves)."""
    file_exists = os.path.exists(EPOCH_LOG_PATH)
    with open(EPOCH_LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "Loss", "Timestamp"])
        writer.writerow([epoch, f"{loss:.6f}", datetime.datetime.now().strftime("%H:%M:%S")])

def load_chunk(files):
    """Loads a list of files into RAM safely."""
    df_list = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if 'target_action' in df.columns:
                df_list.append(df)
        except: pass
    
    if not df_list: return None, None
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 1. Label Cleaning
    action_map = {'FOLD': 0, 'fold': 0, 'call': 1, 'CALL': 1, 'check': 1, 'CHECK': 1, 'bet': 2, 'BET': 2, 'raise': 2, 'RAISE': 2}
    def map_action(val):
        if isinstance(val, (int, float, np.integer, np.floating)): return int(val)
        return action_map.get(str(val).upper(), None)

    full_df['label'] = full_df['target_action'].apply(map_action)
    full_df = full_df.dropna(subset=['label'])
    
    # 2. Feature Selection (Auto-Detect)
    named_cols = ['pot_odds', 'spr', 'position', 'street', 'current_pot', 'to_call', 'has_pair',
                  'hole_rank_1', 'hole_suit_1', 'hole_rank_2', 'hole_suit_2',
                  'board_rank_1', 'board_suit_1', 'board_rank_2', 'board_suit_2',
                  'board_rank_3', 'board_suit_3', 'board_rank_4', 'board_suit_4',
                  'board_rank_5', 'board_suit_5']
    generic_cols = [f"f{i}" for i in range(21)]

    if all(c in full_df.columns for c in named_cols):
        final_cols = named_cols
        rank_cols = [c for c in named_cols if 'rank' in c]
        suit_cols = [c for c in named_cols if 'suit' in c]
        full_df[rank_cols] /= 13.0
        full_df[suit_cols] /= 4.0
    elif all(c in full_df.columns for c in generic_cols):
        final_cols = generic_cols
    else:
        final_cols = named_cols
        for c in final_cols:
            if c not in full_df.columns: full_df[c] = 0.0

    X = full_df[final_cols].values.astype(np.float32)
    y = full_df['label'].values.astype(np.int64)
    
    return X, y

def train():
    print(f"[INFO] STARTING GRANDMASTER TRAINING (CHUNKED & ROBUST)")
    start_time = time.time()
    
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    if not files:
        print("[ERR] No data found.")
        return

    print(f"[INFO] Found {len(files)} files. Processing in chunks of {CHUNK_SIZE}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PokerNet(INPUT_SIZE, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = StandardScaler()
    
    # PRE-PASS: FIT SCALER
    print("[INFO] Calibrating Scaler on first chunk...")
    X_sample, _ = load_chunk(files[:CHUNK_SIZE])
    if X_sample is None or len(X_sample) == 0:
        print("[ERR] First chunk was empty or corrupt. Aborting.")
        return
    
    money_indices = [0, 1, 4, 5] 
    scaler.fit(X_sample[:, money_indices])
    joblib.dump(scaler, SCALER_PATH)
    del X_sample
    gc.collect()

    final_loss = 0.0
    
    # Reset Log File for this run
    if os.path.exists(EPOCH_LOG_PATH): os.remove(EPOCH_LOG_PATH)

    # TRAINING LOOP
    for epoch in range(EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")
        np.random.shuffle(files)
        total_epoch_loss = 0
        chunks_processed = 0
        
        for i in range(0, len(files), CHUNK_SIZE):
            chunk_files = files[i : i + CHUNK_SIZE]
            X, y = load_chunk(chunk_files)
            if X is None: continue
            
            X[:, money_indices] = scaler.transform(X[:, money_indices])
            tensor_x = torch.from_numpy(X)
            tensor_y = torch.from_numpy(y)
            
            dataset = TensorDataset(tensor_x, tensor_y)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            batch_loss_accum = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                batch_loss_accum += loss.item()
            
            if len(loader) > 0:
                chunk_loss = batch_loss_accum / len(loader)
                total_epoch_loss += chunk_loss
                chunks_processed += 1
                sys.stdout.write(f"\r   -> Processed Chunk {i//CHUNK_SIZE + 1} | Loss: {chunk_loss:.4f}")
                sys.stdout.flush()
            
            del X, y, tensor_x, tensor_y, dataset, loader
            gc.collect()
            
        if chunks_processed > 0:
            final_loss = total_epoch_loss / chunks_processed
            print(f"\n   -> Epoch Average Loss: {final_loss:.4f}")
            log_epoch(epoch+1, final_loss) # Log detailed curve
        
        torch.save(model.state_dict(), MODEL_PATH)

    duration = time.time() - start_time
    print(f"\n[SUCCESS] GRANDMASTER SAVED: {MODEL_PATH}")
    log_training_run("Grandmaster_Robust", len(files), EPOCHS, final_loss, duration)

if __name__ == "__main__":
    train()