import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from collections import deque
from treys import Card, Evaluator, Deck

### --- CONFIGURATION --- ###
PROJECT_ROOT = "/home/roman/github/Poker-AI"
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")
CHAMPION_PATH = os.path.join(MODELS_ROOT, "league_champion.pth")
SELF_PLAY_DIR = os.path.join(MODELS_ROOT, "self_play_evolution")
os.makedirs(SELF_PLAY_DIR, exist_ok=True)

# Hyperparameters
# Set to 5000 so the script finishes occasionally, allowing the League to run
TOTAL_EPISODES = 5000 
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 10000
MEMORY_SIZE = 20000
TARGET_UPDATE = 10
OPPONENT_UPDATE = 1000

### --- 1. NETWORK --- ###
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

### --- 2. SELF-PLAY ENVIRONMENT --- ###
class SelfPlayEnv:
    def __init__(self, opponent_model, device):
        self.opponent = opponent_model
        self.device = device
        self.evaluator = Evaluator()
        self.deck = Deck()
        
    def reset(self):
        self.deck = Deck()
        self.hero_hand = self.deck.draw(2)
        self.villain_hand = self.deck.draw(2)
        self.community_cards = []
        self.pot = 150 # SB+BB
        self.hero_stack = 10000 - 50 
        self.villain_stack = 10000 - 100 
        self.current_bet = 100
        self.hero_bet = 50
        self.villain_bet = 100
        self.street = 0 
        self.done = False
        
        return self.get_state_vector(is_hero=True)

    def get_state_vector(self, is_hero=True):
        hand = self.hero_hand if is_hero else self.villain_hand
        board = self.community_cards
        
        def grs(c_int):
            if c_int is None: return 0, 0
            return Card.get_rank_int(c_int), Card.get_suit_int(c_int)

        h_r1, h_s1 = grs(hand[0])
        h_r2, h_s2 = grs(hand[1])
        b_feats = []
        for i in range(5):
            c = board[i] if i < len(board) else None
            r, s = grs(c)
            b_feats.extend([r, s])

        my_bet = self.hero_bet if is_hero else self.villain_bet
        to_call = self.current_bet - my_bet
        stack = self.hero_stack if is_hero else self.villain_stack
        
        pot_odds = to_call / (self.pot + to_call) if (self.pot + to_call) > 0 else 0
        spr = stack / self.pot if self.pot > 0 else 0
        
        vec = [
            pot_odds, spr, 0.5, len(board), self.pot, to_call, 0,
            h_r1, h_s1, h_r2, h_s2,
            *b_feats
        ]
        return np.array(vec, dtype=np.float32)

    def step(self, action_idx):
        # HERO ACTION
        if action_idx == 0: # FOLD
            return self.get_state_vector(), -50, True 
            
        elif action_idx == 1: # CALL
            to_call = self.current_bet - self.hero_bet
            self.hero_stack -= to_call
            self.hero_bet += to_call
            self.pot += to_call
            
        elif action_idx == 2: # RAISE
            raise_amt = (self.current_bet - self.hero_bet) + 100
            self.hero_stack -= raise_amt
            self.hero_bet += raise_amt
            self.pot += raise_amt
            self.current_bet = self.hero_bet

        # VILLAIN ACTION (AI)
        v_state = self.get_state_vector(is_hero=False)
        with torch.no_grad():
            t = torch.tensor(np.array([v_state]), device=self.device)
            v_action = torch.argmax(self.opponent(t), dim=1).item()
            
        if v_action == 0: # Villain Folds
            return self.get_state_vector(), self.pot, True 
            
        elif v_action == 2: # Villain Raises
            to_call = (self.current_bet - self.villain_bet) + 100
            self.villain_stack -= to_call
            self.villain_bet += to_call
            self.pot += to_call
            self.current_bet = self.villain_bet
            
            # Hero forced Call to simplify training loop
            hero_call = self.current_bet - self.hero_bet
            self.hero_stack -= hero_call
            self.pot += hero_call
            
        else: # Villain Calls
            call_amt = self.current_bet - self.villain_bet
            self.villain_stack -= call_amt
            self.pot += call_amt

        # NEXT STREET / SHOWDOWN
        if self.street < 3:
            self.street += 1
            needed = [0, 3, 1, 1][self.street]
            self.community_cards.extend(self.deck.draw(needed))
            return self.get_state_vector(), 0, False
        else:
            h_score = self.evaluator.evaluate(self.community_cards, self.hero_hand)
            v_score = self.evaluator.evaluate(self.community_cards, self.villain_hand)
            
            if h_score < v_score:
                return self.get_state_vector(), self.pot, True
            else:
                return self.get_state_vector(), -self.pot, True

### --- 3. TRAINING LOOP --- ###
def train_self_play():
    print(f"[INFO] Starting Self-Play (Hero vs Champion)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = PokerNet().to(device)
    target_net = PokerNet().to(device)
    opponent_net = PokerNet().to(device)
    
    if os.path.exists(CHAMPION_PATH):
        print(f"[INFO] Loading Champion: {CHAMPION_PATH}")
        weights = torch.load(CHAMPION_PATH, map_location=device)
        policy_net.load_state_dict(weights)
        target_net.load_state_dict(weights)
        opponent_net.load_state_dict(weights)
        opponent_net.eval()
    else:
        print("[ERR] No Champion found! Run train_models.py first.")
        return

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    memory = deque(maxlen=MEMORY_SIZE)
    env = SelfPlayEnv(opponent_net, device)
    
    steps = 0
    wins = 0
    
    for episode in range(TOTAL_EPISODES):
        state = env.reset()
        done = False
        
        while not done:
            if random.random() > EPSILON_START:
                with torch.no_grad():
                    t = torch.tensor(np.array([state]), device=device)
                    action = torch.argmax(policy_net(t), dim=1).item()
            else:
                action = random.randint(0, 2)
            
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            if done and reward > 0: wins += 1
            
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_state, b_action, b_reward, b_next, b_done = zip(*batch)
                
                b_state = torch.tensor(np.array(b_state), device=device, dtype=torch.float32)
                b_action = torch.tensor(b_action, device=device, dtype=torch.long).unsqueeze(1)
                b_reward = torch.tensor(b_reward, device=device, dtype=torch.float32)
                b_next = torch.tensor(np.array(b_next), device=device, dtype=torch.float32)
                
                q_curr = policy_net(b_state).gather(1, b_action)
                q_next = target_net(b_next).max(1)[0].detach()
                q_targ = b_reward + (GAMMA * q_next)
                
                loss = nn.SmoothL1Loss()(q_curr, q_targ.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # EVOLUTION CHECKPOINT
        if episode > 0 and episode % OPPONENT_UPDATE == 0:
            new_champ = os.path.join(SELF_PLAY_DIR, f"evolution_{random.randint(1000,9999)}.pth")
            torch.save(policy_net.state_dict(), new_champ)
            print(f"[SAVE] New candidate saved: {new_champ}")

    # Final Save
    final_path = os.path.join(SELF_PLAY_DIR, f"evolution_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print("[INFO] Training Session Complete.")

if __name__ == "__main__":
    train_self_play()