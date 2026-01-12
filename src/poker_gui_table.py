import threading
import queue
import time
import sys
import os
import io
import re
import random
import csv
import datetime
import FreeSimpleGUI as sg
from PIL import Image, ImageDraw, ImageFont
from treys import Card

from poker_gym import PokerTable, Player, RandomBot, ConservativeBot, ManiacBot, CallingStationBot, SmartBot, EquityBot, hand_to_str

### --- 1. PATH CONFIGURATION --- ###
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ASSET_ROOT = os.path.join(PROJECT_ROOT, "assets")
DATA_DIR = os.path.join(PROJECT_ROOT, "data") 
CARDS_DIR = os.path.join(ASSET_ROOT, "poker_cards_assets")
BACKS_DIR = os.path.join(ASSET_ROOT, "poker_backcard_assets")
CHIPS_DIR = os.path.join(ASSET_ROOT, "poker_chips_assets")
REPO_RES_DIR = os.path.join(PROJECT_ROOT, "notebooks/Poker/resources")
TABLE_IMG = os.path.join(REPO_RES_DIR, "table.png")

RAW_W, RAW_H = 46, 62
SCALE_FACTOR = 2
CARD_W, CARD_H = RAW_W * SCALE_FACTOR, RAW_H * SCALE_FACTOR
CURRENT_BACK_IMG = None

def pick_new_deck_color():
    global CURRENT_BACK_IMG
    try:
        valid_backs = [f for f in os.listdir(BACKS_DIR) if f.endswith('.png')]
        if valid_backs:
            selected = random.choice(valid_backs)
            CURRENT_BACK_IMG = os.path.join(BACKS_DIR, selected)
        else:
            CURRENT_BACK_IMG = os.path.join(BACKS_DIR, "back_0_0.png")
    except Exception: pass
pick_new_deck_color()

### --- 2. DATA RECORDING (SNAPSHOT) --- ###
class GameRecorder:
    def __init__(self, filename="poker_session_log.csv"):
        if not os.path.exists(DATA_DIR):
            try: os.makedirs(DATA_DIR)
            except: pass
        self.filepath = os.path.join(DATA_DIR, filename)
        self.hand_count = 0
        base_headers = ["Hand_ID", "Timestamp", "Winner", "Pot_Size", "Winning_Rank", "Board", "Showdown_Hands"]
        player_headers = []
        for i in range(1, 7): 
            p = f"P{i}"
            player_headers.extend([f"{p}_Name", f"{p}_Stack", f"{p}_Net", f"{p}_BuyIns"])
        self.headers = base_headers + player_headers
        try:
            if not os.path.exists(self.filepath):
                with open(self.filepath, mode='w', newline='') as f:
                    csv.writer(f).writerow(self.headers)
            sys.__stdout__.write(f"[INFO] Snapshot Logger initialized.\n")
        except Exception as e:
            sys.__stdout__.write(f"[CRITICAL ERROR]: Could not create CSV file: {e}\n")

    def record_full_snapshot(self, winner, pot, rank, board, showdown_str, player_stats_str):
        self.hand_count += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [self.hand_count, timestamp, winner, pot, rank, board, showdown_str]
        
        # Parse Player Stats String (joined by ||)
        # Format: Name|Stack|Net|BuyIns || Name2...
        if player_stats_str:
            all_players = player_stats_str.split(" || ")
            for p_str in all_players:
                # p_str: "Seat 1 (Bot)|1000|-50|1"
                parts = p_str.split('|')
                if len(parts) == 4:
                    row.extend(parts)
                else:
                    row.extend(["N/A", 0, 0, 0])
        
        # Pad columns
        while len(row) < len(self.headers): row.append("")
        
        try:
            with open(self.filepath, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            sys.__stdout__.write(f"[ERROR] Writing Snapshot: {e}\n")

### --- 3. GUI LOGIC & PARSER --- ####
CHIP_MAP = {
    'white':  {'val': 1,    'row': 1, 'col_start': 0},
    'red':    {'val': 5,    'row': 0, 'col_start': 0},
    'blue':   {'val': 10,   'row': 0, 'col_start': 4},
    'green':  {'val': 25,   'row': 2, 'col_start': 0},
    'black':  {'val': 100,  'row': 3, 'col_start': 0},
    'purple': {'val': 500,  'row': 1, 'col_start': 4},
    'gold':   {'val': 1000, 'row': 3, 'col_start': 4},
    'pink':   {'val': 5000, 'row': 2, 'col_start': 4},
}
def get_chip_image_data(amount):
    if amount <= 0: return None
    if amount < 5: color = 'white'
    elif amount < 10: color = 'red'
    elif amount < 25: color = 'blue'
    elif amount < 100: color = 'green'
    elif amount < 500: color = 'black'
    elif amount < 1000: color = 'purple'
    elif amount < 5000: color = 'gold'
    else: color = 'pink'
    cfg = CHIP_MAP[color]
    ratio = amount / cfg['val']
    offset = 0
    if ratio >= 10: offset = 3
    elif ratio >= 5: offset = 2
    elif ratio >= 2: offset = 1
    target_row = cfg['row']
    target_col = cfg['col_start'] + offset
    filename = f"chip_{target_row}_{target_col}.png"
    path = os.path.join(CHIPS_DIR, filename)
    if os.path.exists(path):
        try:
            img = Image.open(path).convert("RGBA")
            img = img.resize((30 * SCALE_FACTOR, 30 * SCALE_FACTOR), Image.Resampling.NEAREST)
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            return bio.getvalue()
        except: return None
    return create_fallback_chip(color)

def get_button_img(name):
    filename = f"button_{name}.png"
    path = os.path.join(CHIPS_DIR, filename)
    if os.path.exists(path):
        try:
            img = Image.open(path).convert("RGBA")
            img = img.resize((30, 30), Image.Resampling.NEAREST)
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            return bio.getvalue()
        except: return None
    return None

def create_fallback_chip(color_name):
    size = (40, 40)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([(0,0), (39,39)], fill=color_name, outline="white", width=2)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def get_card_data(card_int=None):
    if card_int is None: path = CURRENT_BACK_IMG
    else: path = os.path.join(CARDS_DIR, f"{Card.int_to_str(card_int)}.png")
    if path is None or not os.path.exists(path): return None
    try:
        img = Image.open(path).convert("RGBA")
        img = img.resize((CARD_W, CARD_H), Image.Resampling.NEAREST)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception: return None

def get_table_bg(width, height):
    if os.path.exists(TABLE_IMG):
        img = Image.open(TABLE_IMG).resize((width, height), Image.Resampling.LANCZOS)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    return None

class GUIPlayer(Player):
    def __init__(self, name, stack=1000, action_queue=None):
        super().__init__(name, stack)
        self.action_queue = action_queue
    def action(self, current_table_bet, pot, community_cards):
        print(f"\n>>> ACTION TO: {self.name}") 
        return self.action_queue.get()

class GUILogger:
    def __init__(self, window, recorder=None, table=None, hero=None):
        self.window = window
        self.recorder = recorder
        self.table = table
        self.hero = hero
        self.real_stdout = sys.__stdout__

    def write(self, text):
        try: self.window.write_event_value('-LOG-', text)
        except: pass 
        
        clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text).strip()
        
        # PARSE: [CSV_DATA] HAND^Winner^Pot^Rank^Board^Showdown^Stats
        if "[CSV_DATA] HAND" in clean_text and self.recorder:
            try:
                raw_data = clean_text.replace("[CSV_DATA] HAND^", "")
                
                # Split by CARET (^) 
                # Winner(0), Pot(1), Rank(2), Board(3), Showdown(4), Stats(5)
                parts = raw_data.split('^')
                
                if len(parts) >= 6:
                    winner = parts[0]
                    pot = parts[1]
                    rank = parts[2]
                    board = parts[3]
                    showdown = parts[4]
                    stats_str = parts[5]
                    
                    self.recorder.record_full_snapshot(winner, pot, rank, board, showdown, stats_str)
                    
            except Exception as e:
                # self.real_stdout.write(f"[LOG ERROR]: {e}\n")
                pass

    def flush(self): pass

def run_poker_table():
    sg.theme('DarkGreen')
    WIN_W, WIN_H = 1000, 700
    CENTER_X, CENTER_Y = WIN_W // 2, WIN_H // 2
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
        [graph],
        [sg.Multiline(size=(120, 6), key='-LOGBOX-', autoscroll=True, disabled=True, font=("Courier", 10))],
        [sg.Text("Action:", font=("Helvetica", 12)), 
         sg.Button('FOLD', key='-FOLD-', button_color=('white', 'red'), size=(10,2), disabled=True),
         sg.Button('CALL', key='-CALL-', button_color=('white', 'blue'), size=(10,2), disabled=True),
         sg.Button('RAISE', key='-RAISE-', button_color=('black', 'orange'), size=(10,2), disabled=True),
         sg.Input('50', key='-AMT-', size=(6,1)),
         sg.Push(), sg.Button('EXIT')]
    ]
    window = sg.Window('Poker AI Gym', layout, finalize=True)
    if get_table_bg(WIN_W - 50, WIN_H - 120): graph.draw_image(data=get_table_bg(WIN_W - 50, WIN_H - 120), location=(25, WIN_H - 10))
    else: graph.draw_rectangle((50, 50), (WIN_W-50, WIN_H-150), fill_color='darkgreen')
    pot_text_id = graph.draw_text("POT: $0", (CENTER_X, CENTER_Y - 30), color="yellow", font=("Helvetica", 24, "bold"))

    POSITIONS = {
        'Community': [(CENTER_X - 160, CENTER_Y + 60), (CENTER_X - 80, CENTER_Y + 60), 
                      (CENTER_X, CENTER_Y + 60), (CENTER_X + 80, CENTER_Y + 60), (CENTER_X + 160, CENTER_Y + 60)],
        'Hero':  (CENTER_X, 220), 'Seat1': (120, CENTER_Y + 60), 'Seat2': (180, 580),
        'Seat3': (CENTER_X, 600), 'Seat4': (WIN_W - 180, 580), 'Seat5': (WIN_W - 120, CENTER_Y + 60)
    }
    SEAT_MAP = {"Hero": "Hero", "Seat 1": "Seat1", "Seat 2": "Seat2", "Seat 3": "Seat3", "Seat 4": "Seat4", "Seat 5": "Seat5"}
    ui_elements = {'cards': {}, 'labels': {}, 'board': [], 'buttons': []}

    def update_player_info(player_obj, pos_key, is_active=True):
        x, y = POSITIONS[pos_key]
        net = player_obj.stack - player_obj.total_invested
        equity_str = f"({net})"
        label = f"{player_obj.name}\n${player_obj.stack}\n{equity_str}"
        color = "white" if is_active else "gray"
        if pos_key in ui_elements['labels']: graph.delete_figure(ui_elements['labels'][pos_key])
        ui_elements['labels'][pos_key] = graph.draw_text(label, (x + 30, y - CARD_H - 25), color=color, font=("Helvetica", 10, "bold"))

    def draw_hand(pos_key, cards=None, face_down=False):
        if pos_key in ui_elements['cards']:
            for fig in ui_elements['cards'][pos_key]: graph.delete_figure(fig)
        ui_elements['cards'][pos_key] = []
        x, y = POSITIONS[pos_key]
        for i in range(2):
            offset = (i * 40) - 20 
            data = get_card_data(None) if face_down else (get_card_data(cards[i]) if cards else None)
            if data: ui_elements['cards'][pos_key].append(graph.draw_image(data=data, location=(x + offset, y)))

    def draw_board(cards):
        for fig in ui_elements['board']: graph.delete_figure(fig)
        ui_elements['board'] = []
        for i, c_int in enumerate(cards):
            x, y = POSITIONS['Community'][i]
            ui_elements['board'].append(graph.draw_image(data=get_card_data(c_int), location=(x, y)))

    def draw_dealer_buttons(dealer_idx):
        for btn in ui_elements['buttons']: graph.delete_figure(btn)
        ui_elements['buttons'] = []
        n = len(all_players)
        d_pos = SEAT_MAP.get(all_players[dealer_idx].name, "Hero")
        dx, dy = POSITIONS[d_pos]
        d_img = get_button_img("dealer")
        if d_img: ui_elements['buttons'].append(graph.draw_image(data=d_img, location=(dx - 40, dy - 10)))
        sb_idx = (dealer_idx + 1) % n
        s_pos = SEAT_MAP.get(all_players[sb_idx].name, "Hero")
        sx, sy = POSITIONS[s_pos]
        sb_img = get_button_img("sb")
        if sb_img: ui_elements['buttons'].append(graph.draw_image(data=sb_img, location=(sx + 80, sy - 80)))
        bb_idx = (dealer_idx + 2) % n
        b_pos = SEAT_MAP.get(all_players[bb_idx].name, "Hero")
        bx, by = POSITIONS[b_pos]
        bb_img = get_button_img("bb")
        if bb_img: ui_elements['buttons'].append(graph.draw_image(data=bb_img, location=(bx + 80, by - 80)))

    def animate_bet(pos_key, amount):
        start_x, start_y = POSITIONS[pos_key]
        start_x += 30; start_y -= 40
        chip_data = get_chip_image_data(amount) or get_chip_image_data(5)
        chip_id = graph.draw_image(data=chip_data, location=(start_x, start_y))
        steps, end_x, end_y = 10, CENTER_X, CENTER_Y - 30
        dx, dy = (end_x - start_x) / steps, (end_y - start_y) / steps
        for _ in range(steps):
            graph.move_figure(chip_id, dx, dy)
            time.sleep(0.01)
            window.refresh()
        graph.delete_figure(chip_id)

    ### --- SETUP --- ####
    action_queue = queue.Queue()
    hero = GUIPlayer("Hero", stack=1000, action_queue=action_queue)
    recorder = GameRecorder()
    available_personalities = [CallingStationBot, ManiacBot, ConservativeBot, RandomBot, SmartBot, EquityBot]
    bots = []
    for i in range(1, 6):
        bot = random.choice(available_personalities)(f"Seat {i}")
        bots.append(bot)
        sys.__stdout__.write(f"[DEBUG]: Seat {i} is {bot.__class__.__name__}\n")
    all_players = [hero] + bots
    table = PokerTable(all_players)
    sys.stdout = GUILogger(window, recorder, table, hero)

    def game_thread():
        while True:
            if hero.stack <= 0: pass 
            pick_new_deck_color()
            window.write_event_value('-RESET-', '')
            table.play_hand()
            time.sleep(2)

    t = threading.Thread(target=game_thread, daemon=True)
    t.start()

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'EXIT'): break
        
        if event == '-LOG-':
            text = values[event]
            window['-LOGBOX-'].update(text, append=True)
            if any(x in text for x in ["calls", "RAISES", "posts", "buys in"]):
                graph.delete_figure(pot_text_id)
                pot_text_id = graph.draw_text(f"POT: ${table.pot}", (CENTER_X, CENTER_Y-30), color="yellow", font=("Helvetica", 24, "bold"))
                amt_match = re.search(r"\$(\d+)", text)
                amt = int(amt_match.group(1)) if amt_match else 5
                for p in all_players:
                    if p.name in text:
                        pos = SEAT_MAP.get(p.name, "Hero")
                        if "buys in" not in text: animate_bet(pos, amt)
                        update_player_info(p, pos)
                        break
            if any(x in text for x in ["[FLOP]","[TURN]","[RIVER]"]): draw_board(table.community_cards)
            if len(hero.hand) == 2: draw_hand('Hero', hero.hand)

        if event == '-RESET-':
            for p in all_players: update_player_info(p, SEAT_MAP.get(p.name, "Hero"))
            draw_board([])
            for p in bots: draw_hand(SEAT_MAP[p.name], face_down=True)
            draw_hand('Hero', cards=[]) 
            draw_dealer_buttons(table.dealer_idx)

        if event == '-LOG-' and "ACTION TO: Hero" in values[event]:
            window['-FOLD-'].update(disabled=False)
            call_amt = table.current_table_bet - hero.current_bet
            window['-CALL-'].update(text=f"CALL ${call_amt}", disabled=False)
            window['-RAISE-'].update(disabled=False)

        if event == '-FOLD-': action_queue.put(('fold', 0)); disable_buttons(window)
        elif event == '-CALL-': 
            amt = table.current_table_bet - hero.current_bet
            action_queue.put(('call', amt)); disable_buttons(window)
        elif event == '-RAISE-':
            try:
                amt = int(values['-AMT-'])
                total = (table.current_table_bet - hero.current_bet) + amt
                action_queue.put(('raise', total)); disable_buttons(window)
            except: pass

    window.close()

def disable_buttons(window):
    window['-FOLD-'].update(disabled=True)
    window['-CALL-'].update(disabled=True)
    window['-RAISE-'].update(disabled=True)

if __name__ == "__main__":
    run_poker_table()