import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime
import random
import glob
from collections import deque
from treys import Card

### --- IMPORT FROM NEW HEADLESS FILE --- ###
from poker_headless import PokerEnv

### --- CONFIGURATION --- ###
PROJECT_ROOT = "/home/roman/github/Poker-AI"
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(MODELS_ROOT, f"rl_{TIMESTAMP}")
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY = 2000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

### --- 1. THE BRAIN --- ###
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

    def forward(self, x):
        return self.network(x)

### --- 2. EXPERIENCE REPLAY --- ###
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
 
### --- 3. UTILS --- ###
def get_state_vector(state_dict):
    hole = state_dict.get('hole_cards', [])
    board = state_dict.get('board_cards', [])
    
    def grs(c_int):
        if c_int is None: return 0, 0
        return Card.get_rank_int(c_int), Card.get_suit_int(c_int)

    h_r1, h_s1 = grs(hole[0] if len(hole)>0 else None)
    h_r2, h_s2 = grs(hole[1] if len(hole)>1 else None)
    
    b_feats = []
    for i in range(5):
        c = board[i] if i < len(board) else None
        r, s = grs(c)
        b_feats.extend([r,s])

    vec = [
        state_dict.get('pot_odds', 0),
        state_dict.get('spr', 0),
        state_dict.get('position', 0.5),
        state_dict.get('street', 0),
        state_dict.get('current_pot', 0),
        state_dict.get('to_call', 0),
        state_dict.get('has_pair', 0),
        h_r1, h_s1, h_r2, h_s2,
        *b_feats
    ]
    return np.array(vec, dtype=np.float32)

### --- 4. MAIN LOOP --- ###
def train_rl():
    print(f"[INFO] Starting Headless Reinforcement Learning")
    print(f"[INFO] Saving versions to: {SAVE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = PokerNet().to(device)
    target_net = PokerNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Attempt to load latest supervised model
    try:
        list_of_dirs = glob.glob(os.path.join(MODELS_ROOT, 'v_*'))
        if list_of_dirs:
            latest_dir = max(list_of_dirs, key=os.path.getctime)
            latest_file = os.path.join(latest_dir, "poker_brain.pth")
            if os.path.exists(latest_file):
                policy_net.load_state_dict(torch.load(latest_file, map_location=device))
                target_net.load_state_dict(policy_net.state_dict())
                print(f"[INFO] Loaded pre-trained brain: {latest_file}")
    except Exception as e:
        print(f"[WARN] Starting from scratch. Error: {e}")

    env = PokerEnv()
    steps_done = 0
    
    for i_episode in range(10000):
        state_dict = env.reset_game()
        state = get_state_vector(state_dict)
        
        done = False
        
        while not done:
            sample = random.random()
            eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                np.exp(-1. * steps_done / EPSILON_DECAY)
            steps_done += 1
            
            # Select Action
            if sample > eps_threshold:
                with torch.no_grad():
                    st_tensor = torch.tensor([state], device=device)
                    q_values = policy_net(st_tensor)
                    action_idx = q_values.max(1)[1].item()
            else:
                action_idx = random.randrange(3)
            
            action_map = {0: 'fold', 1: 'call', 2: 'raise'}
            cmd = action_map[action_idx]
            
            # Step Environment
            next_state_dict, reward, done = env.step(0, cmd)
            next_state = get_state_vector(next_state_dict)
            
            # Memory and Train
            memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next, batch_done = zip(*transitions)
                
                b_state = torch.tensor(np.array(batch_state), device=device, dtype=torch.float32)
                b_action = torch.tensor(batch_action, device=device, dtype=torch.long).unsqueeze(1)
                b_reward = torch.tensor(batch_reward, device=device, dtype=torch.float32)
                b_next = torch.tensor(np.array(batch_next), device=device, dtype=torch.float32)
                
                current_q = policy_net(b_state).gather(1, b_action)
                next_q = target_net(b_next).max(1)[0].detach()
                expected_q = b_reward + (GAMMA * next_q)
                
                loss = nn.SmoothL1Loss()(current_q, expected_q.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if i_episode % 100 == 0:
            print(f"[INFO] Episode {i_episode} complete.")

        if i_episode % 500 == 0:
            p = os.path.join(SAVE_DIR, f"poker_rl_{i_episode}.pth")
            torch.save(policy_net.state_dict(), p)
            print(f"[SAVE] Checkpoint saved: {p}")

if __name__ == "__main__":
    train_rl()