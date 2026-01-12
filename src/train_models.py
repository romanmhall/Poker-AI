import torch
import torch.nn as nn
import numpy as np
import os
import glob
import random
import shutil
import time
from treys import Card, Evaluator, Deck

### --- CONFIGURATION --- ###
PROJECT_ROOT = "/home/roman/github/Poker-AI"
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")
CHAMPION_PATH = os.path.join(MODELS_ROOT, "league_champion.pth")

HANDS_TO_PLAY = 1000
START_STACK = 10000
BIG_BLIND = 100

### --- 1. NETWORK ARCHITECTURE --- ###
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

### --- 2. THE AGENT --- ###
class Agent:
    def __init__(self, filepath, name, device):
        self.name = name
        self.filepath = filepath
        self.device = device
        self.model = PokerNet().to(device)
        self.stack = START_STACK
        self.winnings = 0
        self.active = True
        
        # Load Weights
        try:
            self.model.load_state_dict(torch.load(filepath, map_location=device))
            self.model.eval() # Inference Mode
            self.working = True
        except Exception as e:
            print(f"[ERR] Failed to load {name}: {e}")
            self.working = False

    def decide(self, state_vector):
        if not self.working: return 1 # Default Call
        with torch.no_grad():
            t = torch.tensor(np.array([state_vector]), device=self.device)
            out = self.model(t)
            return torch.argmax(out, dim=1).item() # 0=Fold, 1=Call, 2=Raise
        
###--- 3. TOURNAMENT ENGINE ---###
class LeagueTournament:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agents = []
        self.evaluator = Evaluator()
        
    def recruit_agents(self):
        print(f"[LEAGUE] Scouting for models in {MODELS_ROOT}...")
        
        all_models = []
        # Recursively find all .pth files
        for root, dirs, files in os.walk(MODELS_ROOT):
            for file in files:
                if file.endswith(".pth") and ("checkpoint" in file or "brain" in file or "rl" in file):
                    full_path = os.path.join(root, file)
                    # Create a readable ID
                    parent = os.path.basename(root)
                    name = f"{parent}/{file}"
                    # Store tuple: (path, name, timestamp)
                    all_models.append((full_path, name, os.path.getmtime(full_path)))

        # Sort by newest first
        all_models.sort(key=lambda x: x[2], reverse=True)
        
        # Take Top 6 Unique
        candidates = all_models[:6]
        
        if len(candidates) == 0:
            print("[ERR] No models found! Run training first.")
            return False

        print(f"\n[LEAGUE] The Contenders:")
        for path, name, _ in candidates:
            print(f"   [CONTENDER] {name}")
            self.agents.append(Agent(path, name, self.device))
            
        # If fewer than 6, fill with clones of the newest
        while len(self.agents) < 6:
            clone_source = self.agents[0]
            print(f"   + Cloning {clone_source.name} to fill seat.")
            self.agents.append(Agent(clone_source.filepath, f"Clone_{len(self.agents)}", self.device))
            
        return True

    def get_state_vector(self, agent_idx, community_cards, pot, to_call):
        agent = self.agents[agent_idx]
        
        def grs(c_int):
            if c_int is None: return 0, 0
            return Card.get_rank_int(c_int), Card.get_suit_int(c_int)

        h_r1, h_s1 = grs(agent.hand[0])
        h_r2, h_s2 = grs(agent.hand[1])
        
        b_feats = []
        for i in range(5):
            c = community_cards[i] if i < len(community_cards) else None
            r, s = grs(c)
            b_feats.extend([r, s])
            
        # Basic Math
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        spr = agent.stack / pot if pot > 0 else 0
        
        vec = [
            pot_odds, spr, 0.5, len(community_cards), pot, to_call, 0,
            h_r1, h_s1, h_r2, h_s2,
            *b_feats
        ]
        return np.array(vec, dtype=np.float32)

    def run_hand(self):
        deck = Deck()
        pot = 0
        community_cards = []
        active_indices = [i for i, a in enumerate(self.agents) if a.stack > 0]
        
        if len(active_indices) < 2: return 
        
        # Deal
        for i in active_indices:
            self.agents[i].hand = deck.draw(2)
            self.agents[i].active = True
            self.agents[i].current_bet = 0
        
        # Blinds
        sb_idx = active_indices[0]
        bb_idx = active_indices[1]
        
        self.agents[sb_idx].stack -= 50; self.agents[sb_idx].current_bet = 50
        self.agents[bb_idx].stack -= 100; self.agents[bb_idx].current_bet = 100
        pot = 150
        current_bet = 100
        
        ### --- STREET LOOP --- ###
        streets = [0, 3, 4, 5]
        for street_cards in streets:
            # Deal Board
            if street_cards > 0:
                needed = street_cards - len(community_cards)
                if needed > 0: community_cards.extend(deck.draw(needed))
            
            # Betting Loop (Simplified: 1 raise allowed per street to prevent infinite loops)
            raises_this_street = 0
            
            for i in active_indices:
                agent = self.agents[i]
                if not agent.active or agent.stack <= 0: continue
                
                to_call = current_bet - agent.current_bet
                state = self.get_state_vector(i, community_cards, pot, to_call)
                action = agent.decide(state) # 0=Fold, 1=Call, 2=Raise
                
                if action == 0 and to_call > 0: # FOLD
                    agent.active = False
                elif action == 2 and raises_this_street < 2: # RAISE (Limit 2 raises)
                    raises_this_street += 1
                    raise_amt = to_call + BIG_BLIND
                    if raise_amt > agent.stack: raise_amt = agent.stack # All in
                    agent.stack -= raise_amt
                    agent.current_bet += raise_amt
                    pot += raise_amt
                    current_bet = agent.current_bet
                else: # CALL
                    if to_call > agent.stack: to_call = agent.stack
                    agent.stack -= to_call
                    agent.current_bet += to_call
                    pot += to_call

            # Survivors?
            survivors = [i for i in active_indices if self.agents[i].active]
            if len(survivors) == 1:
                winner_idx = survivors[0]
                self.agents[winner_idx].stack += pot
                self.agents[winner_idx].winnings += pot
                return

        ### --- SHOWDOWN --- ###
        best_score = 9999
        winners = []
        
        for i in survivors:
            score = self.evaluator.evaluate(community_cards, self.agents[i].hand)
            if score < best_score:
                best_score = score
                winners = [i]
            elif score == best_score:
                winners.append(i)
        
        # Split Pot
        if len(winners) > 0:
            share = pot // len(winners)
            for w in winners:
                self.agents[w].stack += share
                self.agents[w].winnings += share

    def run(self):
        if not self.recruit_agents(): return

        print(f"\n[INFO] STARTING LEAGUE MATCH ({HANDS_TO_PLAY} Hands)")
        print("-" * 50)
        
        start_time = time.time()
        
        for h in range(HANDS_TO_PLAY):
            # Shuffle seats to balance position advantage
            if h % 10 == 0: random.shuffle(self.agents)
            
            # Re-buy if busted
            for a in self.agents:
                if a.stack < BIG_BLIND: a.stack = START_STACK
                
            self.run_hand()
            
            if h % 100 == 0:
                print(f"   Hand {h}/{HANDS_TO_PLAY}...", end="\r")

        duration = time.time() - start_time
        print(f"\n[INFO] MATCH COMPLETE ({duration:.1f}s)")
        
        # Results
        print("\n=== LEADERBOARD ===")
        print(f"{'AGENT':<45} | {'WIN RATE (BB/100)':<20}")
        print("-" * 70)
        
        results = sorted(self.agents, key=lambda x: x.winnings, reverse=True)
        
        for p in results:
            bb_100 = (p.winnings / BIG_BLIND) / (HANDS_TO_PLAY / 100) / 6 
            print(f"{p.name:<45} | {bb_100:.2f}")

        champion = results[0]
        print("-" * 70)
        print(f"[CHAMPION] {champion.name}")
        
        # PROMOTION LOGIC
        print(f"\n[LEAGUE] Promoting Champion to: {CHAMPION_PATH}")
        try:
            shutil.copyfile(champion.filepath, CHAMPION_PATH)
            print("[SUCCESS] league_champion.pth updated.")
        except Exception as e:
            print(f"[ERR] Promotion failed: {e}")

if __name__ == "__main__":
    t = LeagueTournament()
    t.run()