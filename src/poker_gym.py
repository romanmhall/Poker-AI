import random
import time
import re
import os
import json
from treys import Card, Evaluator, Deck

try:
    from poker_ai_interface import PokerAI
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

### --- CONFIGURATION --- ####
PREFLOP_FILE = "data/preflop_lookup.json"
SB_AMOUNT = 5
BB_AMOUNT = 10
STARTING_STACK = 1000

### --- VISUALIZATION & UTILS --- ###
def card_to_str(card_int):
    return f"[{Card.int_to_str(card_int)}]"

def hand_to_str(hand_ints):
    return " ".join([card_to_str(c) for c in hand_ints])

def get_hand_type(hand_ints):
    if len(hand_ints) != 2: return None
    c1, c2 = hand_ints[0], hand_ints[1]
    r1 = Card.get_rank_int(c1); r2 = Card.get_rank_int(c2)
    s1 = Card.get_suit_int(c1); s2 = Card.get_suit_int(c2)
    rank_map = {0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'T', 9:'J', 10:'Q', 11:'K', 12:'A'}
    if r1 < r2: r1, r2 = r2, r1; s1, s2 = s2, s1
    char1 = rank_map[r1]; char2 = rank_map[r2]
    if r1 == r2: return f"{char1}{char2}"
    elif s1 == s2: return f"{char1}{char2}s"
    else: return f"{char1}{char2}o"

def get_log_name(player):
    return f"{player.name} ({player.__class__.__name__})"

### --- 2. PLAYER CLASSES --- ###
class Player:
    def __init__(self, name, stack=STARTING_STACK):
        self.name = name
        self.stack = stack
        self.starting_stack = stack 
        self.hand = []
        self.folded = False
        self.current_bet = 0
        self.all_in = False
        self.total_invested = stack
        self.buy_in_count = 1

    @property
    def net_equity(self):
        return self.stack - self.total_invested

    def rebuy(self):
        self.stack += self.starting_stack
        self.total_invested += self.starting_stack
        self.buy_in_count += 1
        print(f"[REBUY] {self.name} buys in for ${self.starting_stack}. (Total Invested: ${self.total_invested})")

    def reset_for_hand(self):
        self.hand = []
        self.folded = False
        self.current_bet = 0
        self.all_in = False

    def action(self, current_table_bet, pot, community_cards):
        raise NotImplementedError

class RandomBot(Player):
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        if to_call > self.stack: to_call = self.stack
        roll = random.random()
        if to_call > 0 and roll < 0.10: return 'fold', 0
        if roll < 0.30: 
            raise_amt = to_call + random.randint(10, 50)
            return 'raise', raise_amt
        return 'call', to_call

class ConservativeBot(Player):
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        ranks = [Card.get_rank_int(c) for c in self.hand]
        if to_call > 0 and sum(ranks) < 16: return 'fold', 0
        if sum(ranks) > 22: return 'raise', to_call + 50
        return 'call', to_call

class ManiacBot(Player):
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        if random.random() < 0.80: 
            return 'raise', to_call + random.randint(50, 150)
        return 'call', to_call

class CallingStationBot(Player):
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        if to_call > (self.stack * 0.5): return 'fold', 0
        return 'call', to_call

class SmartBot(Player):
    def __init__(self, name, stack=STARTING_STACK):
        super().__init__(name, stack)
        self.brain = PokerAI() if AI_AVAILABLE else None

    def action(self, current_table_bet, pot, community_cards):
        if not self.brain: return 'call', current_table_bet - self.current_bet
        to_call = current_table_bet - self.current_bet
        total_pot = pot + current_table_bet
        pot_odds = to_call / (total_pot + to_call) if (total_pot + to_call) > 0 else 0
        spr = self.stack / total_pot if total_pot > 0 else 0
        h_ranks = [Card.get_rank_int(c) for c in self.hand]
        has_pair = 1 if len(h_ranks) == 2 and h_ranks[0] == h_ranks[1] else 0
        game_state = {
            'pot_odds': pot_odds, 'spr': spr, 'position': 0.5, 'street': len(community_cards),
            'current_pot': total_pot, 'to_call': to_call, 'has_pair': has_pair,
            'hole_cards': self.hand, 'board_cards': community_cards
        }
        decision = self.brain.decide(game_state)
        if decision == "FOLD": return 'fold', 0
        elif decision == "RAISE": return 'raise', to_call + max(50, int(total_pot * 0.5))
        return 'call', to_call

class EquityBot(Player):
    def __init__(self, name, stack=STARTING_STACK):
        super().__init__(name, stack)
        self.preflop_table = {}
        self.evaluator = Evaluator()
        self.load_memory()
    def load_memory(self):
        if os.path.exists(PREFLOP_FILE):
            try:
                with open(PREFLOP_FILE, 'r') as f: self.preflop_table = json.load(f)
            except: pass
    def calculate_postflop_equity(self, community_cards, iterations=200):
        wins = 0; deck = Deck()
        known = self.hand + community_cards
        deck.cards = [c for c in deck.cards if c not in known]
        needed = 5 - len(community_cards)
        if needed == 0: return 0 
        for _ in range(iterations):
            random.shuffle(deck.cards)
            villain = deck.cards[:2]
            runout = deck.cards[2:2+needed]
            full_board = community_cards + runout
            our_score = self.evaluator.evaluate(full_board, self.hand)
            villain_score = self.evaluator.evaluate(full_board, villain)
            if our_score < villain_score: wins += 1
            elif our_score == villain_score: wins += 0.5
        return wins / iterations
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        total_pot = pot + current_table_bet
        pot_odds = to_call / (total_pot + to_call) if (total_pot + to_call) > 0 else 0
        equity = 0.0
        if not community_cards:
            hand_type = get_hand_type(self.hand)
            equity = self.preflop_table.get(hand_type, 0.50)
        else:
            equity = self.calculate_postflop_equity(community_cards)
        margin = 0.05
        if equity < (pot_odds + margin) and to_call > 0: return 'fold', 0
        if equity > 0.70: return 'raise', to_call + int(total_pot * 0.75)
        return 'call', to_call

### --- 3. POKER ENGINE --- ###
class PokerTable:
    def __init__(self, players):
        self.players = players
        self.deck = Deck()
        self.evaluator = Evaluator()
        self.community_cards = []
        self.pot = 0
        self.current_table_bet = 0
        self.dealer_idx = 0 

    def post_blinds(self):
        n = len(self.players)
        sb_idx = (self.dealer_idx + 1) % n
        bb_idx = (self.dealer_idx + 2) % n
        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]
        sb_amt = min(sb_player.stack, SB_AMOUNT)
        sb_player.stack -= sb_amt
        sb_player.current_bet = sb_amt
        self.pot += sb_amt
        print(f"{sb_player.name} posts SB ${sb_amt}")
        bb_amt = min(bb_player.stack, BB_AMOUNT)
        bb_player.stack -= bb_amt
        bb_player.current_bet = bb_amt
        self.pot += bb_amt
        print(f"{bb_player.name} posts BB ${bb_amt}")
        self.current_table_bet = BB_AMOUNT

    def betting_round(self, street_name):
        print(f"\n--- {street_name} Betting ---")
        n = len(self.players)
        if street_name == "Pre-Flop": start_idx = (self.dealer_idx + 3) % n
        else: start_idx = (self.dealer_idx + 1) % n
        betting_active = True
        players_to_act = {i: True for i in range(n) if not self.players[i].folded and not self.players[i].all_in}
        loops = 0
        while betting_active and players_to_act and loops < 5:
            loops += 1
            idx_order = [ (start_idx + i) % n for i in range(n) ]
            action_occured = False
            for i in idx_order:
                p = self.players[i]
                if p.folded or p.all_in: continue
                if p.current_bet == self.current_table_bet and i not in players_to_act: continue
                if i in players_to_act: del players_to_act[i]
                try: act, amt = p.action(self.current_table_bet, self.pot, self.community_cards)
                except: act, amt = 'fold', 0 
                if act == 'fold':
                    p.folded = True
                    print(f"{p.name} folds.")
                elif act == 'call':
                    to_call = self.current_table_bet - p.current_bet
                    spend = min(p.stack, to_call)
                    p.stack -= spend
                    self.pot += spend
                    p.current_bet += spend
                    if p.stack == 0: p.all_in = True
                    print(f"{p.name} calls ${spend}.")
                elif act == 'raise':
                    needed = amt - p.current_bet
                    spend = min(p.stack, needed)
                    p.stack -= spend
                    self.pot += spend
                    p.current_bet += spend
                    if p.current_bet > self.current_table_bet:
                        self.current_table_bet = p.current_bet
                        players_to_act = {x: True for x in range(n) if x != i and not self.players[x].folded and not self.players[x].all_in}
                        action_occured = True
                    if p.stack == 0: p.all_in = True
                    print(f"{p.name} RAISES to ${p.current_bet}!")
            if not action_occured and not players_to_act: betting_active = False
        for p in self.players: p.current_bet = 0
        self.current_table_bet = 0

    def play_hand(self):
        for p in self.players:
            if p.stack <= 0: p.rebuy()
        self.deck.shuffle()
        self.community_cards = []
        self.pot = 0
        self.current_table_bet = 0
        self.dealer_idx = (self.dealer_idx + 1) % len(self.players)
        for p in self.players: p.reset_for_hand()
        print("\n" + "="*40 + "\n NEW HAND\n" + "="*40)
        self.post_blinds()
        for p in self.players: p.hand = self.deck.draw(2)
        self.betting_round("Pre-Flop")
        active_count = sum(1 for p in self.players if not p.folded)
        if active_count > 1:
            self.community_cards = self.deck.draw(3)
            print(f"\n[FLOP]: {hand_to_str(self.community_cards)}")
            self.betting_round("Flop")
        if active_count > 1 and sum(1 for p in self.players if not p.folded) > 1:
            self.community_cards += self.deck.draw(1)
            print(f"\n[TURN]: {hand_to_str(self.community_cards)}")
            self.betting_round("Turn")
        if active_count > 1 and sum(1 for p in self.players if not p.folded) > 1:
            self.community_cards += self.deck.draw(1)
            print(f"\n[RIVER]: {hand_to_str(self.community_cards)}")
            self.betting_round("River")
        self.resolve_winner()

    def resolve_winner(self):
        active = [p for p in self.players if not p.folded]
        board_str = hand_to_str(self.community_cards) if self.community_cards else "No Board"
        winner_names = "Unknown"
        winner_rank = "Fold"
        showdown_data = []

        if len(active) == 1:
            winner = active[0]
            print(f"Everyone folded. {winner.name} wins ${self.pot}!")
            winner.stack += self.pot
            winner_names = get_log_name(winner)
        else:
            best_score = 10000
            winners = []
            print(f"Board: {board_str}\n")
            for p in active:
                score = self.evaluator.evaluate(self.community_cards, p.hand)
                rank_class = self.evaluator.get_rank_class(score)
                rank_str = self.evaluator.class_to_string(rank_class)
                hand_str = hand_to_str(p.hand)
                log_name = get_log_name(p)
                showdown_data.append(f"{log_name}:{hand_str} ({rank_str})")
                print(f"{p.name}: {hand_str} ({rank_str})")
                if score < best_score:
                    best_score = score
                    winners = [p]
                    winner_rank = rank_str
                elif score == best_score:
                    winners.append(p)
            win_share = int(self.pot / len(winners)) if winners else 0
            winner_names = " & ".join([get_log_name(w) for w in winners])
            for w in winners: w.stack += win_share
            display_names = " & ".join([w.name for w in winners])
            print(f"\n[WINNER]: {display_names} takes ${self.pot} with {winner_rank}!")

        # Build Safe Strings using Caret (^) as Main Separator
        player_stats = []
        for p in self.players:
            # Stats: Name|Stack|Net|BuyIns (using | internal separator is fine here)
            p_data = f"{get_log_name(p)}|{p.stack}|{p.net_equity}|{p.buy_in_count}"
            player_stats.append(p_data)
        
        # Join stats with DOUBLE PIPE (||) to avoid confusion with internal single pipe
        full_stats_str = " || ".join(player_stats)
        hands_str = " || ".join(showdown_data) if showdown_data else "N/A"

        # MAIN SEPARATOR IS NOW ^ (Caret)
        print(f"[CSV_DATA] HAND^{winner_names}^{self.pot}^{winner_rank}^{board_str}^{hands_str}^{full_stats_str}")