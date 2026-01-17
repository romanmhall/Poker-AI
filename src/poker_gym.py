import random
import time
from treys import Card, Evaluator, Deck

try:
    from poker_ai_interface import PokerAI
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# CONFIGURATION
STARTING_STACK = 1000

# UTILS
def card_to_str(card_int):
    return f"[{Card.int_to_str(card_int)}]"

def hand_to_str(hand_ints):
    return " ".join([card_to_str(c) for c in hand_ints])

# PLAYER CLASSES
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

    def reset_for_hand(self):
        self.hand = []
        self.folded = False
        self.current_bet = 0
        self.all_in = False

    def action(self, current_table_bet, pot, community_cards):
        raise NotImplementedError

# BOT PERSONALITIES
class RandomBot(Player):
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        return 'call', to_call

class ManiacBot(Player):
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        # Raises aggressively
        return 'raise', to_call + 100
        
class CallingStationBot(Player):
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        return 'call', to_call

class ConservativeBot(Player):
    """Folds unless it's cheap to call."""
    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        if to_call > 20: 
            return 'fold', 0
        return 'call', to_call

class EquityBot(Player):
    """Calculates basic hand strength."""
    def __init__(self, name, stack=STARTING_STACK):
        super().__init__(name, stack)
        self.evaluator = Evaluator()

    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        
        # Simple logic: Call if we have a pair or better
        if len(community_cards) >= 3:
            score = self.evaluator.evaluate(community_cards, self.hand)
            if score < 6000: # Decent hand
                return 'call', to_call
            if to_call > 0:
                return 'fold', 0
        
        return 'call', to_call

# THE SMART BOT (WITH EXPLORATION)
class SmartBot(Player):
    def __init__(self, name, stack=STARTING_STACK, epsilon=0.0):
        super().__init__(name, stack)
        self.brain = PokerAI() if AI_AVAILABLE else None
        self.epsilon = epsilon # 0.0 = Pure AI, 0.2 = 20% Random Chaos

    def action(self, current_table_bet, pot, community_cards):
        to_call = current_table_bet - self.current_bet
        total_pot_chips = pot + current_table_bet
        
        # 1. EXPLORATION (addresses statemates)
        if self.epsilon > 0 and random.random() < self.epsilon:
            moves = ['fold', 'call']
            if self.stack > to_call: moves.append('raise')
            
            random_act = random.choice(moves)
            if random_act == 'fold': return 'fold', 0
            if random_act == 'call': return 'call', to_call
            if random_act == 'raise': 
                return 'raise', to_call + max(100, int(total_pot_chips * 0.5))

        # 2. STANDARD AI LOGIC
        if not self.brain: return 'call', to_call

        game_state = {
            'pot': total_pot_chips,
            'to_call': to_call,
            'stack': self.stack,
            'street': len(community_cards),
            'position': 0.5,
            'hole_cards': self.hand,
            'board_cards': community_cards
        }
        
        decision = self.brain.decide(game_state)
        
        # 3. FILTERS (Free Roll Logic)
        if decision == "FOLD":
            if to_call <= 0: return 'call', 0 # Always check if free
            return 'fold', 0

        if decision == "RAISE":
            raise_amt = to_call + max(100, int(total_pot_chips * 0.5))
            return 'raise', raise_amt
        
        return 'call', to_call

# POKER ENGINE
class PokerTable:
    def __init__(self, players):
        self.players = players
        self.deck = Deck()
        self.evaluator = Evaluator()
        self.community_cards = []
        self.pot = 0
        self.current_table_bet = 0
        self.dealer_idx = 0 

    def play_hand(self):
        for p in self.players: p.reset_for_hand()
        self.deck.shuffle()
        self.community_cards = []
        self.pot = 0
        self.current_table_bet = 0
        
        active_players = [p for p in self.players if p.stack > 0]
        if len(active_players) < 2: return 

        for p in active_players:
            p.hand = self.deck.draw(2)

        sb_idx = (self.dealer_idx + 1) % len(active_players)
        bb_idx = (self.dealer_idx + 2) % len(active_players)
        
        sb_p = active_players[sb_idx]
        bb_p = active_players[bb_idx]
        
        sb_p.stack -= 50; sb_p.current_bet = 50; self.pot += 50
        bb_p.stack -= 100; bb_p.current_bet = 100; self.pot += 100
        self.current_table_bet = 100

        def run_betting_round():
            active = [p for p in active_players if not p.folded and p.stack > 0]
            if len(active) < 2: return
            
            for p in active:
                if p.folded or p.all_in: continue
                act, amt = p.action(self.current_table_bet, self.pot, self.community_cards)
                
                if act == 'fold': p.folded = True
                elif act == 'call':
                    if amt > p.stack: amt = p.stack
                    p.stack -= amt; p.current_bet += amt; self.pot += amt
                elif act == 'raise':
                    if amt > p.stack: amt = p.stack
                    p.stack -= amt; p.current_bet += amt; self.pot += amt
                    if p.current_bet > self.current_table_bet:
                        self.current_table_bet = p.current_bet

        run_betting_round() # Pre-Flop
        
        for street_cards in [3, 1, 1]: 
            if len([p for p in active_players if not p.folded]) < 2: break
            self.community_cards.extend(self.deck.draw(street_cards))
            self.current_table_bet = 0 
            for p in active_players: p.current_bet = 0
            run_betting_round()

        self.resolve_winner(active_players)
        self.dealer_idx = (self.dealer_idx + 1) % len(active_players)

    def resolve_winner(self, active_players):
        active = [p for p in active_players if not p.folded]
        if not active: return

        board_str = hand_to_str(self.community_cards) if self.community_cards else "PreFlop"
        
        if len(active) == 1:
            winner = active[0]
            winner.stack += self.pot
            winner_rank = "Fold"
            hands_str = f"{winner.name}: Wins (Fold)"
            winners = [winner]
        else:
            best_score = 10000
            winners = []
            showdown_data = []

            for p in active:
                score = self.evaluator.evaluate(self.community_cards, p.hand)
                rank_str = self.evaluator.class_to_string(self.evaluator.get_rank_class(score))
                showdown_data.append(f"{p.name}:{hand_to_str(p.hand)} ({rank_str})")
                if score < best_score:
                    best_score = score
                    winners = [p]
                    winner_rank = rank_str
                elif score == best_score:
                    winners.append(p)
            
            hands_str = " || ".join(showdown_data)
            win_share = self.pot // len(winners)
            for w in winners: w.stack += win_share
        
        player_stats = [f"{p.name}|{p.stack}|{p.net_equity}|{p.buy_in_count}" for p in self.players]
        full_stats_str = " || ".join(player_stats)
        
        print(f"[CSV_DATA] HAND^{winners[0].name}^{self.pot}^{winner_rank}^{board_str}^{hands_str}^{full_stats_str}")