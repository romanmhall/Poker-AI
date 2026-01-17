import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
import time
import csv
import datetime
import numpy as np

# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
LOG_FILE = os.path.join(PROJECT_ROOT, "data", "training_history.csv")

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

class PokerDataset(Dataset):
    def __init__(self, files):
        self.data = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                if 'target_action' not in df.columns: continue
                self.data.append(df)
            except: pass
        
        if self.data:
            self.full_df = pd.concat(self.data)
            
            # SANITIZE TARGET ACTIONS
            action_map = {
                'FOLD': 0, 'Fold': 0, 'fold': 0,
                'CALL': 1, 'Call': 1, 'call': 1, 'CHECK': 1, 'Check': 1,
                'RAISE': 2, 'Raise': 2, 'raise': 2, 'BET': 2, 'Bet': 2, 'ALL-IN': 2
            }

            def clean_action(val):
                if isinstance(val, (int, float, np.integer, np.floating)):
                    return int(val)
                if isinstance(val, str):
                    val = val.upper().strip()
                    return action_map.get(val, None)
                return None

            self.full_df['target_action'] = self.full_df['target_action'].apply(clean_action)
            self.full_df = self.full_df.dropna(subset=['target_action'])
            
            # Extract Features
            feature_cols = [c for c in self.full_df.columns if c.startswith('f') or 'hole' in c or 'board' in c or 'pot' in c or 'spr' in c or 'position' in c]
            feature_cols = feature_cols[:21]

            if len(feature_cols) < 21:
                feature_cols = self.full_df.columns[:21]

            self.X = torch.tensor(self.full_df[feature_cols].values, dtype=torch.float32)
            self.y = torch.tensor(self.full_df['target_action'].values.astype(int), dtype=torch.long)
        else:
            self.X = torch.empty(0)
            self.y = torch.empty(0)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def log_training_run(model_name, source_files, epochs, final_loss, duration):
    """Writes the training result to the CSV log."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model_Name", "Source_Data", "Epochs", "Final_Loss", "Duration_Sec"])
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sources = f"{len(source_files)} Files" 
        writer.writerow([timestamp, model_name, sources, epochs, f"{final_loss:.4f}", f"{duration:.1f}"])

def train_champion(model_path, data_files, epochs=2):
    if not data_files:
        print("[TRAIN] No data provided.")
        return

    start_time = time.time()
    
    # 1. Load Data
    dataset = PokerDataset(data_files)
    if len(dataset) == 0:
        print("[TRAIN] Dataset empty (files might be corrupt or incompatible).")
        return
        
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PokerNet().to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            print("[TRAIN] Warning: Could not load existing weights. Starting fresh.")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # 3. Training Loop
    final_loss = 0
    print(f"   [STUDY] Training on {len(dataset)} hands for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count if count > 0 else 0
        final_loss = avg_loss
    
    # 4. Save
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True) # <--- THE FIX
        
    torch.save(model.state_dict(), model_path)
    
    # 5. Log
    duration = time.time() - start_time
    log_training_run("League_Champion", data_files, epochs, final_loss, duration)
    print(f"   [STUDY] Complete. Loss: {final_loss:.4f}")

if __name__ == "__main__":
    all_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "training_ready", "**", "*.parquet"), recursive=True)
    champ_path = os.path.join(MODELS_DIR, "league_champion.pth")
    train_champion(champ_path, all_files, epochs=5)