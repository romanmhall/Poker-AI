import torch
import numpy as np
import pandas as pd
import os
import random
import glob
import time
from treys import Card, Evaluator, Deck
from train_models import PokerNet

# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROSTER_DIR = os.path.join(PROJECT_ROOT, "data", "models", "active_roster")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "training_ready", "self_play")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# POKER CONSTANTS
BIG_BLIND = 20
START_STACK = 2000

class SelfPlayEngine:
    def __init__(self):
        self.evaluator = Evaluator()
        self.deck = Deck()
        self.opponents = self.load_roster()
        
    def load_roster(self):
        files = glob.glob(os.path.join(ROSTER_DIR, "*.pth"))
        agents = []
        if not files:
            print("[WARN] No roster found. Creating random opponent.")
            net = PokerNet()
            agents.append(net)
        else:
            selected = random.sample(files, min(len(files), 5))
            for f in selected:
                net = PokerNet()
                try:
                    net.load_state_dict(torch.load(f, map_location='cpu'))
                    net.eval()
                    agents.append(net)
                except: pass
            print(f"[SELF-PLAY] Loaded {len(agents)} opponents.")
        return agents

    def get_state_vector(self, hand, community, pot, to_call, stack):
        def grs(c): return (Card.get_rank_int(c), Card.get_suit_int(c)) if c else (0,0)
        h_r1, h_s1 = grs(hand[0]); h_r2, h_s2 = grs(hand[1])
        b_feats = []
        for i in range(5):
            c = community[i] if i < len(community) else None
            r, s = grs(c)
            b_feats.extend([r, s])
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        spr = stack / pot if pot > 0 else 0
        vec = [pot_odds, spr, 0.5, len(community), pot, to_call, 0,
               h_r1, h_s1, h_r2, h_s2, *b_feats]
        return np.array(vec, dtype=np.float32)

    def play_batch(self, batch_size=5000):
        data = []
        
        for _ in range(batch_size):
            self.deck.shuffle()
            hero_hand = self.deck.draw(2)
            opp_hands = [self.deck.draw(2) for _ in self.opponents]
            community = self.deck.draw(5) 
            
            # Setup Pot
            pot = 30 # Small + Big Blind
            hero_stack = START_STACK
            to_call = 20
            
            # 1. STATE
            state = self.get_state_vector(hero_hand, [], pot, to_call, hero_stack)
            
            # 2. ACTION (Hero picks a move)
            if self.opponents:
                hero_net = random.choice(self.opponents)
                if random.random() < 0.20: # Exploration
                    action = random.choice([0, 1, 2])
                else:
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(state).unsqueeze(0)
                        action = torch.argmax(hero_net(state_tensor)).item()
            else:
                action = random.choice([0, 1, 2])

            # 3. CALCULATE PROFIT (BB/100 Logic)
            # Simulate the hand result to see how many BBs we win or lose.
            
            hero_score = self.evaluator.evaluate(community, hero_hand)
            opp_scores = [self.evaluator.evaluate(community, oh) for oh in opp_hands]
            best_opp_score = min(opp_scores)
            
            did_win = hero_score < best_opp_score
            
            # BB PROFIT CALCULATION
            # Fold: lose nothing (0 BB), or small blind if we posted it.
            # Call/Raise and Lose: lose the bet amount (-BB).
            # Call/Raise and Win: win the Pot (+BB).
            
            target_label = action 
            
            if action == 0: # FOLD
                # Folding a winner = loss of opportunity cost
                if did_win:
                    # Lost the whole pot that could have been won
                    profit_bb = - (pot / BIG_BLIND) 
                    target_label = 2 # Should have Raised (Aggressive)
                else:
                    # Saving money = good fold
                    profit_bb = 0.5 # Small reward for discipline
                    target_label = 0
            
            elif action == 1: # CALL
                if did_win:
                    profit_bb = (pot / BIG_BLIND)
                    target_label = 1
                else:
                    profit_bb = -1.0 # Lost 1 BB (the Call)
                    target_label = 0 # Should have Folded
            
            elif action == 2: # RAISE
                if did_win:
                    # Large reward for building pot and winning
                    profit_bb = (pot / BIG_BLIND) * 1.5 
                    target_label = 2
                else:
                    # Expensive mistake.
                    profit_bb = -2.0 # Lost 2 BB (the Raise)
                    target_label = 0 # Should have Folded

            # SAVE DATA
            # Avoiding win/loss data and utilizing proft margin
            # If the profit/loss impact was significant (> 0.5 BB), keep and learn from it.
            if abs(profit_bb) >= 0.5:
                row = list(state) + [target_label]
                data.append(row)

        # Save to Parquet
        if not data: return
        
        cols = [f"f{i}" for i in range(21)] + ["target_action"]
        df = pd.DataFrame(data, columns=cols)
        
        timestamp = int(time.time() * 1000)
        filename = os.path.join(OUTPUT_DIR, f"self_play_{timestamp}.parquet")
        df.to_parquet(filename)
        
        print(f"[GEN] Created {len(data)} hands (BB/100 Weighted).")

if __name__ == "__main__":
    engine = SelfPlayEngine()
    engine.play_batch()