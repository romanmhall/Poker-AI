import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime
import random
import glob
from collections import deque
from treys import Card, Evaluator, Deck
from poker_headless import PokerEnv

### --- CONFIGURATION --- ###
PROJECT_ROOT = "/home/roman/github/Poker-AI"
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(MODELS_ROOT, f"rl_smart_{TIMESTAMP}")
os.makedirs(SAVE_DIR, exist_ok=True)

# Training Hyperparameters
TOTAL_EPISODES = 100000     
TEST_INTERVAL = 1000        
TEST_HANDS = 500            
PATIENCE_LIMIT = 20         

# RL Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY = 5000
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

#### --- 2. REPLAY MEMORY --- ###
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

### --- 4. VALIDATION FUNCTION --- ###
def run_validation(policy_net, device):
    """Runs a quick Gauntlet without training to measure skill"""
    evaluator = Evaluator()
    chips_won = 0
    wins = 0
    hands = 0
    big_blind = 100
    
    policy_net.eval()
    
    for _ in range(TEST_HANDS):
        deck = Deck()
        hero_hand = deck.draw(2)
        villain_hand = deck.draw(2)
        board = deck.draw(5)
        
        current_board = board[:3]
        pot = big_blind * 2
        
        state_dict = {
            'hole_cards': hero_hand,
            'board_cards': current_board,
            'pot_odds': 0.33, 'spr': 5.0, 'position': 0.5, 'street': 3,
            'current_pot': pot, 'to_call': 0, 'has_pair': 0
        }
        state_vec = get_state_vector(state_dict)
        
        with torch.no_grad():
            t_state = torch.tensor(np.array([state_vec]), device=device)
            output = policy_net(t_state)
            action_idx = torch.argmax(output, dim=1).item()
            
        if action_idx == 0: 
            chips_won -= 50
        else:
            h_score = evaluator.evaluate(board, hero_hand)
            v_score = evaluator.evaluate(board, villain_hand)
            if h_score < v_score:
                chips_won += pot
                wins += 1
            else:
                chips_won -= big_blind
        hands += 1
        
    policy_net.train() 
    bb_100 = (chips_won / big_blind) / (hands / 100)
    return bb_100, wins

### --- 5. MAIN LOOP --- ###
def train_and_validate():
    print(f"[INFO] Starting Smart Training (Early Stopping Enabled)")
    print(f"[INFO] Output: {SAVE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = PokerNet().to(device)
    target_net = PokerNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    memory = ReplayMemory(MEMORY_SIZE)
    
    ### --- FIXED LOADING LOGIC --- ###
    try:
        list_of_dirs = glob.glob(os.path.join(MODELS_ROOT, 'v_*')) + glob.glob(os.path.join(MODELS_ROOT, 'rl_*'))
        latest_file = None

        if list_of_dirs:
            # Sort by modification time to find the newest folder
            list_of_dirs.sort(key=os.path.getmtime, reverse=True)
            
            for latest_dir in list_of_dirs:
                # Try to find a valid model file in this directory
                if "rl_" in latest_dir:
                    files = glob.glob(os.path.join(latest_dir, "*.pth"))
                    if files: 
                        latest_file = max(files, key=os.path.getmtime)
                        break # Found one, stop searching
                else:
                    temp_path = os.path.join(latest_dir, "poker_brain.pth")
                    if os.path.exists(temp_path):
                        latest_file = temp_path
                        break # Found one, stop searching
            
            if latest_file and os.path.exists(latest_file):
                policy_net.load_state_dict(torch.load(latest_file, map_location=device))
                target_net.load_state_dict(policy_net.state_dict())
                print(f"[INFO] Loaded Brain: {latest_file}")
            else:
                 print("[INFO] No valid .pth files found in recent folders. Starting fresh.")
        else:
            print("[INFO] No model directories found. Starting fresh.")

    except Exception as e:
        print(f"[WARN] Error during model loading (Starting fresh): {e}")
    # ---------------------------

    env = PokerEnv()
    steps_done = 0
    
    best_winrate = -9999
    patience_counter = 0
    
    for i_episode in range(1, TOTAL_EPISODES + 1):
        state_dict = env.reset_game()
        state = get_state_vector(state_dict)
        done = False
        
        while not done:
            sample = random.random()
            eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                np.exp(-1. * steps_done / EPSILON_DECAY)
            steps_done += 1
            
            if sample > eps_threshold:
                with torch.no_grad():
                    st_tensor = torch.tensor(np.array([state]), device=device)
                    q_values = policy_net(st_tensor)
                    action_idx = q_values.max(1)[1].item()
            else:
                action_idx = random.randrange(3)
            
            action_map = {0: 'fold', 1: 'call', 2: 'raise'}
            cmd = action_map[action_idx]
            
            next_state_dict, reward, done = env.step(0, cmd)
            next_state = get_state_vector(next_state_dict)
            
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
            
        if i_episode % TEST_INTERVAL == 0:
            print(f"\n[TEST] Validating at Episode {i_episode}...")
            bb_100, wins = run_validation(policy_net, device)
            
            print(f"   Win Rate: {bb_100:+.2f} BB/100  (Wins: {wins}/{TEST_HANDS})")
            
            ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_{i_episode}.pth")
            torch.save(policy_net.state_dict(), ckpt_path)
            
            if bb_100 > best_winrate:
                best_winrate = bb_100
                patience_counter = 0 
                
                best_path = os.path.join(SAVE_DIR, "best_model.pth")
                torch.save(policy_net.state_dict(), best_path)
                print(f"   [NEW RECORD] Best model saved. Patience reset.")
            else:
                patience_counter += 1
                print(f"   [STAGNATION] No improvement. Patience: {patience_counter}/{PATIENCE_LIMIT}")
                
            if patience_counter >= PATIENCE_LIMIT:
                print(f"\n[STOP] Early Stopping triggered! No improvement for {PATIENCE_LIMIT} cycles.")
                print(f"Best Win Rate achieved: {best_winrate:+.2f} BB/100")
                break
            
            print(f"   Resuming training...")

if __name__ == "__main__":
    train_and_validate()