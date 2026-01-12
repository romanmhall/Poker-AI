import random
from treys import Card, Evaluator, Deck

class Player:
    def __init__(self, name, stack, is_bot=True):
        self.name = name
        self.stack = stack
        self.hand = []
        self.current_bet = 0
        self.is_bot = is_bot
        self.active = True
        self.all_in = False

class PokerEnv:
    def __init__(self, num_players=6, small_blind=50, big_blind=100, stack_size=10000):
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.start_stack = stack_size
        self.evaluator = Evaluator()
        self.players = []
        self.reset_table()

    def reset_table(self):
        """Clears the table and creates new players."""
        self.players = [Player(f"Bot_{i}", self.start_stack) for i in range(self.num_players)]
        self.dealer_idx = 0

    def reset_game(self):
        """Prepares a new hand."""
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        
        # Reset players for new hand
        for p in self.players:
            p.hand = []
            p.current_bet = 0
            p.active = (p.stack > 0)
            p.all_in = False
        
        if sum(p.active for p in self.players) < 2:
            self.reset_table() # Restart if everyone is broke

        # Deal Hands
        for p in self.players:
            if p.active:
                p.hand = self.deck.draw(2)

        # Blinds
        sb_idx = (self.dealer_idx + 1) % len(self.players)
        bb_idx = (self.dealer_idx + 2) % len(self.players)
        
        self.post_blind(self.players[sb_idx], self.small_blind)
        self.post_blind(self.players[bb_idx], self.big_blind)
        
        self.dealer_idx = (self.dealer_idx + 1) % len(self.players)
        return self.get_state_dict(0) # Return state for first player

    def post_blind(self, player, amount):
        if player.stack >= amount:
            player.stack -= amount
            player.current_bet = amount
            self.pot += amount
            self.current_bet = max(self.current_bet, amount)
        else:
            # All-in on blind
            self.pot += player.stack
            player.current_bet = player.stack
            player.stack = 0
            player.all_in = True

    def step(self, player_idx, action_str):
        """
        Executes an action for a specific player.
        Returns: (next_state_dict, reward, done)
        """
        player = self.players[player_idx]
        
        if not player.active:
            return self.get_state_dict(player_idx), 0, True

        prev_stack = player.stack

        # Logic
        if action_str == 'fold':
            player.active = False
        elif action_str == 'call':
            to_call = self.current_bet - player.current_bet
            if to_call > player.stack: to_call = player.stack
            player.stack -= to_call
            player.current_bet += to_call
            self.pot += to_call
        elif action_str == 'raise':
            # Simple fixed raise logic
            raise_amt = self.current_bet + self.big_blind 
            to_add = raise_amt - player.current_bet
            if to_add > player.stack: 
                to_add = player.stack # All-in
            
            player.stack -= to_add
            player.current_bet += to_add
            self.pot += to_add
            self.current_bet = player.current_bet

        # Mock Game Progress (Simplified for Speed)
        # Simulate "rest of hand" roughly to get a reward signal.
        reward = player.stack - prev_stack 
        done = not player.active
        
        return self.get_state_dict(player_idx), reward, done

    def get_state_dict(self, player_idx):
        p = self.players[player_idx]
        return {
            'hole_cards': p.hand,
            'board_cards': self.community_cards,
            'pot_odds': 0.3, # simplified placeholder
            'spr': p.stack / self.pot if self.pot > 0 else 0,
            'position': 0.5,
            'street': len(self.community_cards),
            'current_pot': self.pot,
            'to_call': self.current_bet - p.current_bet,
            'has_pair': 0 # simplified
        }