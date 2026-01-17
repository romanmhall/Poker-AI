import sys
import os
import glob
import pandas as pd
import numpy as np
import FreeSimpleGUI as sg
from treys import Card

# CONFIGURATION
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.append(src_dir)

# IMPORT FROM EXPOSED GUI MODULE 
try:
    from poker_gui_table import (
        get_table_bg, get_card_data, get_chip_image_data, 
        draw_hand, draw_board, update_player_info,
        POSITIONS, CENTER_X, CENTER_Y, WIN_W, WIN_H, CARD_H,
        DATA_DIR, GUIPlayer, Player
    )
except ImportError as e:
    print("\n" + "!"*60)
    print(f"[CRITICAL ERROR] Failed to import from 'poker_gui_table'.")
    print(f"Reason: {e}")
    print("!"*60 + "\n")
    sys.exit()

# HELPER: DATA DECODER
RANK_STR = "23456789TJQKA"
SUIT_STR = "shdc" 

def float_to_card(r_norm, s_norm):
    try:
        r_int = int(round(r_norm * 13))
        s_int = int(round(s_norm * 4))
        r_int = max(0, min(12, r_int))
        s_int = max(0, min(3, s_int))
        card_str = f"{RANK_STR[r_int]}{SUIT_STR[s_int]}"
        return Card.new(card_str)
    except:
        return None

def load_random_parquet_file():
    dirs = [
        os.path.join(project_root, "data", "training_ready", "self_play"),
        os.path.join(project_root, "data", "training_ready", "consolidated")
    ]
    files = []
    for d in dirs:
        files.extend(glob.glob(os.path.join(d, "*.parquet")))
    
    if not files: return None
    import random
    f = random.choice(files)
    print(f"[LOAD] Opening {os.path.basename(f)}...")
    return pd.read_parquet(f)

# THE INSPECTOR APP
def run_inspector():
    sg.theme('DarkGreen')
    
    graph = sg.Graph(
        canvas_size=(WIN_W, WIN_H),
        graph_bottom_left=(0, 0),
        graph_top_right=(WIN_W, WIN_H),
        key='-GRAPH-',
        background_color='green',
        enable_events=True,
        drag_submits=True
    )
    
    layout = [
        [sg.Text("GRANDMASTER DATA INSPECTOR", font=("Helvetica", 16, "bold"), text_color="gold", background_color="black")],
        [graph],
        [sg.Text("Target Action:", font=("Helvetica", 12)), 
         sg.Input("...", key='-ACTION-', size=(15,1), readonly=True, text_color="black"),
         sg.Button('PREV HAND', size=(12,2)), 
         sg.Button('NEXT HAND', size=(12,2), bind_return_key=True),
         sg.Button('LOAD NEW FILE', size=(15,2)),
         sg.Button('EXIT')]
    ]
    
    window = sg.Window('Poker AI Inspector', layout, finalize=True)
    
    # Draw Background
    if get_table_bg(WIN_W - 50, WIN_H - 120): 
        graph.draw_image(data=get_table_bg(WIN_W - 50, WIN_H - 120), location=(25, WIN_H - 10))
    else: 
        graph.draw_rectangle((50, 50), (WIN_W-50, WIN_H-150), fill_color='darkgreen')

    # UI Containers
    ui_elements = {'cards': {}, 'labels': {}, 'board': [], 'buttons': []}
    pot_text_id = graph.draw_text("POT: $0", (CENTER_X, CENTER_Y - 30), color="yellow", font=("Helvetica", 24, "bold"))
    
    df = load_random_parquet_file()
    if df is None:
        sg.popup("No parquet files found in data/training_ready!")
        return

    current_idx = 0
    
    def render_snapshot(idx):
        nonlocal pot_text_id
        if idx < 0 or idx >= len(df): return
        
        row = df.iloc[idx]
        
        # 1. DECODE CARDS
        h1 = float_to_card(row.get('hole_rank_1', row.get('f7')), row.get('hole_suit_1', row.get('f8')))
        h2 = float_to_card(row.get('hole_rank_2', row.get('f9')), row.get('hole_suit_2', row.get('f10')))
        hero_hand = [h1, h2] if h1 and h2 else []
        
        board_cards = []
        for i in range(1, 6):
            r = row.get(f'board_rank_{i}')
            s = row.get(f'board_suit_{i}')
            if r is not None and (r > 0.05 or s > 0.05): 
                c = float_to_card(r, s)
                if c: board_cards.append(c)
        
        # 2. UPDATE GUI (Using Shared Functions)
        draw_hand(graph, ui_elements, 'Hero', hero_hand)
        draw_board(graph, ui_elements, board_cards)
        
        # 3. UPDATE POT & STATS
        pot = row.get('current_pot', 0)
        to_call = row.get('to_call', 0)
        
        graph.delete_figure(pot_text_id)
        pot_text_id = graph.draw_text(f"POT: {pot:.1f} BB", (CENTER_X, CENTER_Y - 30), color="yellow", font=("Helvetica", 24, "bold"))
        
        # 4. SHOW TARGET ACTION
        act_map = {0: 'FOLD', 1: 'CALL', 2: 'RAISE'}
        target = int(row['target_action'])
        act_str = act_map.get(target, "???")
        
        color = "white"
        if act_str == "FOLD": color = "#FF9999" 
        if act_str == "CALL": color = "#9999FF" 
        if act_str == "RAISE": color = "#99FF99" 
        
        window['-ACTION-'].update(f"{act_str}", background_color=color)
        window.set_title(f"Inspector - Hand {idx+1}/{len(df)} | To Call: {to_call:.1f} BB")

    render_snapshot(current_idx)

    # EVENT LOOP
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'EXIT'): break
        
        if event == 'NEXT HAND':
            if current_idx < len(df) - 1:
                current_idx += 1
                render_snapshot(current_idx)
        
        if event == 'PREV HAND':
            if current_idx > 0:
                current_idx -= 1
                render_snapshot(current_idx)
                
        if event == 'LOAD NEW FILE':
            new_df = load_random_parquet_file()
            if new_df is not None:
                df = new_df
                current_idx = 0
                render_snapshot(current_idx)

    window.close()

if __name__ == "__main__":
    run_inspector()