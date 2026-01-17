import sys
import os
import csv
import datetime
import pandas as pd
import numpy as np
from poker_gym import PokerTable, SmartBot, RandomBot, ManiacBot, CallingStationBot

# CONFIGURATION
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_READY_DIR = os.path.join(DATA_DIR, "training_ready")
SMART_PLAY_DIR = os.path.join(TRAIN_READY_DIR, "smart_play") 
LOG_FILE = os.path.join(DATA_DIR, "train_smart_data.csv")

TOTAL_HANDS = 50000 
BATCH_SIZE = 500 # Save parquet every 500 hands

# Ensure dirs exist
if not os.path.exists(TRAIN_READY_DIR): os.makedirs(TRAIN_READY_DIR)
if not os.path.exists(SMART_PLAY_DIR): os.makedirs(SMART_PLAY_DIR)

# 1. HUMAN LOGGING (CSV - For Dashboard)
class SmartRecorder:
    def __init__(self, filename=LOG_FILE):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        self.filepath = filename
        
        base_headers = ["Hand_ID", "Timestamp", "Winner", "Pot_Size", "Winning_Rank", "Board", "Showdown_Hands"]
        player_headers = []
        for i in range(1, 7): 
            p = f"P{i}"
            player_headers.extend([f"{p}_Name", f"{p}_Stack", f"{p}_Net", f"{p}_BuyIns"])
        self.headers = base_headers + player_headers

        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                csv.writer(f).writerow(self.headers)

    def save_snapshot(self, hand_id, raw_string):
        try:
            parts = raw_string.split('^')
            if len(parts) < 6: return

            row = [hand_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            row.extend(parts[0:5])
            
            stats_str = parts[5]
            if stats_str:
                all_players = stats_str.split(" || ")
                for p_str in all_players:
                    p_parts = p_str.split('|')
                    if len(p_parts) == 4: row.extend(p_parts)
                    else: row.extend(["N/A", 0, 0, 0])

            while len(row) < len(self.headers): row.append("")
            with open(self.filepath, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
        except Exception: pass

class HeadlessLogger:
    def __init__(self, recorder, original_stdout):
        self.recorder = recorder
        self.terminal = original_stdout
        self.hand_counter = 0

    def write(self, message):

        # 1. CAPTURE DATA (Goes to File, NOT Screen)
        if "[CSV_DATA] HAND" in message:
            clean_msg = message.replace("[CSV_DATA] HAND^", "").strip()
            self.recorder.save_snapshot(self.hand_counter, clean_msg)
            self.hand_counter += 1
            return

        # 2. PRINT STATUS (Only print important updates)
        if "Saving batch" in message or "Error" in message or "STARTING" in message:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self): self.terminal.flush()

# 2. AI MEMORY (Parquet - For Training)
class MemoryBuffer:
    def __init__(self):
        self.current_hand_moves = {} 
        self.collected_data = [] 

    def record_move(self, player_name, state, action):
        if player_name not in self.current_hand_moves:
            self.current_hand_moves[player_name] = []
        self.current_hand_moves[player_name].append((state, action))

    def save_to_disk(self):
        # 1. Initialize list
        flat_data = []
        
        # 2. Flatten data
        for p_name, moves in self.current_hand_moves.items():
            for state, action in moves:
                flat = {
                    'pot_odds': state.get('to_call') / (state.get('pot') + state.get('to_call')) if (state.get('pot')+state.get('to_call')) > 0 else 0,
                    'spr': state.get('stack') / state.get('pot') if state.get('pot') > 0 else 0,
                    'position': 0.5,
                    'street': state.get('street', 0),
                    'current_pot': state.get('pot', 0),
                    'to_call': state.get('to_call', 0),
                    'target_action': action.upper()
                }
                
                # Cards
                from treys import Card
                def grs(c): return (Card.get_rank_int(c), Card.get_suit_int(c)) if c else (0,0)
                
                hole = state.get('hole_cards', [])
                flat['hole_rank_1'], flat['hole_suit_1'] = grs(hole[0] if len(hole)>0 else None)
                flat['hole_rank_2'], flat['hole_suit_2'] = grs(hole[1] if len(hole)>1 else None)
                
                board = state.get('board_cards', [])
                for i in range(5):
                    r, s = grs(board[i] if len(board)>i else None)
                    flat[f'board_rank_{i+1}'] = r
                    flat[f'board_suit_{i+1}'] = s
                
                flat_data.append(flat)

        # 3. Check if empty
        if not flat_data: return
        
        # 4. Clear buffer
        self.current_hand_moves = {}
        
        # 5. Save to Subfolder (Batch)
        df = pd.DataFrame(flat_data)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SMART_PLAY_DIR, f"smart_{timestamp}.parquet")
        
        try:
            df.to_parquet(filename)
        except: pass

# 3. MAIN LOOP
def train():
    recorder = SmartRecorder()
    original_stdout = sys.stdout
    sys.stdout = HeadlessLogger(recorder, original_stdout)
    
    # Setup Parquet Memory
    memory = MemoryBuffer()

    print(f"--- STARTING GYM TRAINING (SILENT MODE) ---")
    print(f"--- Saving CSV to: {LOG_FILE} ---")
    print(f"--- Saving Parquet to: {SMART_PLAY_DIR} ---")

    # Table Setup
    hero = SmartBot("SmartBot_Gym")
    
    # Wrapper for Hero
    original_action = hero.action
    def recording_wrapper(current_table_bet, pot, community_cards, player=hero, og_act=original_action):
        total_pot = pot + current_table_bet
        to_call = current_table_bet - player.current_bet
        state_snapshot = {
            'pot': total_pot, 'to_call': to_call, 'stack': player.stack,
            'street': len(community_cards), 'hole_cards': player.hand,
            'board_cards': community_cards
        }
        decision, amt = og_act(current_table_bet, pot, community_cards)
        memory.record_move(player.name, state_snapshot, decision)
        return decision, amt
    hero.action = recording_wrapper

    opponents = [ManiacBot("Maniac"), RandomBot("Fish"), CallingStationBot("Station"), ManiacBot("Aggro")]
    table = PokerTable([hero] + opponents)

    try:
        # Loop in batches to save files regularly
        hands_played = 0
        while hands_played < TOTAL_HANDS:
            
            # Play a batch
            for _ in range(BATCH_SIZE):
                if hands_played >= TOTAL_HANDS: break
                
                for p in table.players:
                    if p.stack <= 0: p.stack = 1000 
                table.play_hand()
                hands_played += 1
            
            # Save Batch
            print(f" [Gym] Saving batch... ({hands_played}/{TOTAL_HANDS})")
            memory.save_to_disk()
            
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted.")
    finally:
        sys.stdout = original_stdout

if __name__ == "__main__":
    train()