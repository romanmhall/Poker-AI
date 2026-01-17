import os
import random
import torch
import torch.nn as nn
import glob
import shutil
import csv
import datetime
import re
import math
from treys import Card, Evaluator, Deck
import numpy as np

# CONFIGURATION
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_ROOT = os.path.join(PROJECT_ROOT, "data", "models")
ROSTER_DIR = os.path.join(MODELS_ROOT, "active_roster")
CHAMPION_PATH = os.path.join(MODELS_ROOT, "league_champion.pth")
CHAMPION_NAME_FILE = os.path.join(MODELS_ROOT, "champion_name.txt")
HISTORY_FILE = os.path.join(PROJECT_ROOT, "data", "league_history.csv")

HANDS_TO_PLAY = 1000 
START_STACK = 2000
BIG_BLIND = 20

# CPI TARGETS (configured preffered Stats) 
# Scoring is based on distance from these ideals
TARGET_VPIP = 0.25      # Ideal: 25%
TARGET_PFR = 0.75       # Ideal: Raise 75% of played hands
TARGET_AGG = 2.0        # Ideal: Aggression Factor 2.0
TARGET_3BET = 0.09      # Ideal: 9%

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

class Agent:
    def __init__(self, filepath, name):
        self.filepath = filepath
        self.name = name
        self.net = PokerNet()
        self.net.load_state_dict(torch.load(filepath, map_location='cpu'))
        self.net.eval()
        self.stack = START_STACK
        self.winnings = 0
        
        # Stats
        self.hands_dealt = 0
        self.hands_played = 0
        self.raises = 0
        self.postflop_moves = 0
        self.postflop_aggressions = 0 
        self.three_bet_opportunities = 0
        self.three_bets = 0
        self.cpi_score = 0.0 # Continuous Performance Index (CPI)

class LeagueTournament:
    def __init__(self):
        self.evaluator = Evaluator()
        self.deck = Deck()
        self.agents = []
        
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Winner", "Win_Rate_BB100", "CPI_Score", "VPIP", "PFR_Ratio", "Agg_Score", "ThreeBet_Pct"])

    def recruit_agents(self):
        roster_files = glob.glob(os.path.join(ROSTER_DIR, "*.pth"))
        
        champion_file = CHAMPION_PATH
        if os.path.exists(champion_file):
            c_name = "Champion"
            if os.path.exists(CHAMPION_NAME_FILE):
                with open(CHAMPION_NAME_FILE, 'r') as f: c_name = f.read().strip()
            self.agents.append(Agent(champion_file, c_name + " (Champion)"))
        
        needed = 6 - len(self.agents)
        if len(roster_files) < needed:
            print("[WARN] Not enough roster players for a full table.")
            return False
            
        challengers = random.sample(roster_files, needed)
        for f in challengers:
            self.agents.append(Agent(f, os.path.basename(f)))
        return True

    def get_state_vector(self, hand, community, pot, to_call, stack):
        from treys import Card
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

    def run_hand(self, hand_num):
        self.deck.shuffle()
        community = []
        pot = 30
        for a in self.agents:
            a.hand = self.deck.draw(2); a.hands_dealt += 1; a.in_hand = True; a.current_bet = 0
        self.agents[0].stack -= 10; self.agents[0].current_bet = 10
        self.agents[1].stack -= 20; self.agents[1].current_bet = 20
        current_bet = 20
        active_players = [a for a in self.agents]
        
        for agent in active_players:
            if not agent.in_hand: continue
            
            # 3-Bet Opportunity Logic
            is_3bet_opportunity = False
            if current_bet > 20:
                agent.three_bet_opportunities += 1
                is_3bet_opportunity = True
            
            to_call = current_bet - agent.current_bet
            state = self.get_state_vector(agent.hand, community, pot, to_call, agent.stack)
            
            # from_numpy() for speed and unsqueeze(0) to add the batch dimension (1, 21)
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            
            with torch.no_grad(): 
                action = torch.argmax(agent.net(state_tensor)).item()
            
            if action != 0: agent.hands_played += 1
            if action == 2: 
                agent.raises += 1
                if is_3bet_opportunity: agent.three_bets += 1
            
            if action == 0: agent.in_hand = False
            elif action == 2:
                agent.stack -= (to_call + 20); pot += (to_call + 20); current_bet += 20
            else:
                agent.stack -= to_call; pot += to_call
        
        # Showdown Logic
        remaining = [a for a in active_players if a.in_hand]
        if len(remaining) > 1:
            community = self.deck.draw(5)
            scores = []
            for a in remaining: scores.append((self.evaluator.evaluate(community, a.hand), a))
            scores.sort(key=lambda x: x[0])
            winner = scores[0][1]; winner.winnings += pot; winner.stack += pot
        elif len(remaining) == 1:
            remaining[0].winnings += pot; remaining[0].stack += pot

    def calculate_cpi_score(self, agent):
        """Calculates Continuous Performance Index (0.0 - 1.0) based on Bell Curves."""
        vpip = agent.hands_played / agent.hands_dealt if agent.hands_dealt > 0 else 0
        pfr_ratio = agent.raises / agent.hands_played if agent.hands_played > 0 else 0
        agg_score = agent.postflop_aggressions / agent.postflop_moves if agent.postflop_moves > 0 else 0
        three_bet_pct = agent.three_bets / agent.three_bet_opportunities if agent.three_bet_opportunities > 0 else 0.0

        # GAUSSIAN SCORING FUNCTIONS
        # Score = exp( - (actual - target)^2 / (2 * variance^2) )
        
        # 1. VPIP Score (Target 0.25, strictness 0.1)
        s_vpip = math.exp(-((vpip - TARGET_VPIP)**2) / (2 * (0.1**2)))
        
        # 2. PFR Score (Target 0.75, strictness 0.2)
        s_pfr = math.exp(-((pfr_ratio - TARGET_PFR)**2) / (2 * (0.2**2)))
        
        # 3. Aggression Score (Target 2.0, strictness 1.0)
        s_agg = math.exp(-((agg_score - TARGET_AGG)**2) / (2 * (1.0**2)))

        # 4. 3-Bet Score (Target 0.09, strictness 0.04)
        s_3bet = math.exp(-((three_bet_pct - TARGET_3BET)**2) / (2 * (0.04**2)))
        
        # Weighted Average (VPIP is most important)
        cpi = (s_vpip * 0.4) + (s_pfr * 0.2) + (s_agg * 0.2) + (s_3bet * 0.2)
        
        agent.cpi_score = cpi
        
        print(f"\n[CPI AUDIT] {agent.name}")
        print(f"   - VPIP: {vpip:.2f} -> Score: {s_vpip:.2f}")
        print(f"   - PFR:  {pfr_ratio:.2f} -> Score: {s_pfr:.2f}")
        print(f"   - 3Bet: {three_bet_pct:.2f} -> Score: {s_3bet:.2f}")
        print(f"   = TOTAL CPI: {cpi:.4f} / 1.0")
        
        return cpi, vpip, pfr_ratio, agg_score, three_bet_pct

    def update_roster_filenames(self):
        """Renames roster files to include CPI (e.g. Bot_cpi85.pth)"""
        print("\n[ROSTER] Updating CPI Scores...")
        for agent in self.agents:
            if "Champion" in agent.name: continue 
            
            old_path = agent.filepath
            if not os.path.exists(old_path): continue
            
            # Remove old tags (_sX or _cpiXX)
            base_name = os.path.basename(old_path)
            clean_name = re.sub(r'_s\d', '', base_name)
            clean_name = re.sub(r'_cpi\d+', '', clean_name)
            clean_name = clean_name.replace('.pth', '')
            
            # New Name: Bot_cpi85.pth (Score * 100)
            score_int = int(agent.cpi_score * 100)
            new_name = f"{clean_name}_cpi{score_int}.pth"
            new_path = os.path.join(ROSTER_DIR, new_name)
            
            if old_path != new_path:
                try: os.rename(old_path, new_path)
                except: pass

    def log_result(self, winner_name, win_rate, cpi, vpip, pfr, agg, three_bet):
        with open(HISTORY_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                winner_name, f"{win_rate:.2f}", f"{cpi:.4f}",
                f"{vpip:.3f}", f"{pfr:.3f}", f"{agg:.3f}", f"{three_bet:.3f}"
            ])

    def run(self):
        if not self.recruit_agents(): return None
        print(f"\n[INFO] STARTING LEAGUE MATCH ({HANDS_TO_PLAY} Hands)")
        for h in range(HANDS_TO_PLAY):
            if h % 10 == 0: random.shuffle(self.agents)
            for a in self.agents:
                if a.stack < BIG_BLIND: a.stack = START_STACK
            self.run_hand(h)
        
        # Calculate scores for everyone
        for agent in self.agents:
            self.calculate_cpi_score(agent)
            
        # Sort Leaderboard by WINNINGS, but track CPI
        results = sorted(self.agents, key=lambda x: x.winnings, reverse=True)
        champion = results[0]
        bb_100_winner = (champion.winnings / BIG_BLIND) / (HANDS_TO_PLAY / 100) / 6 
        
        print("-" * 85)
        print(f"[WINNER] {champion.name} (CPI: {champion.cpi_score:.3f})")

        self.update_roster_filenames()
        cpi, vpip, pfr, agg, three_bet = self.calculate_cpi_score(champion)
        self.log_result(champion.name, bb_100_winner, cpi, vpip, pfr, agg, three_bet)

        is_worthy = (cpi > 0.60) # Pass threshold
        
        if champion.filepath != CHAMPION_PATH:
            if is_worthy:
                print(f"\n[NEW KING] {champion.name} promoted! (CPI > 0.60)")
                shutil.copyfile(champion.filepath, CHAMPION_PATH)
                with open(CHAMPION_NAME_FILE, 'w') as f: f.write(champion.name)
            else:
                print(f"\n[DENIED] {champion.name} won, but plays like a fish (CPI {cpi:.2f}).")
        
        return {
            "winner_name": champion.name,
            "is_worthy": is_worthy,
            "cpi": cpi
        }

if __name__ == "__main__":
    t = LeagueTournament()
    t.run()