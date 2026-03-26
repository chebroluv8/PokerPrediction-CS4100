"""
Define Poker Environment
"""
import rlcard 
import random

class LimitHoldEmEnv():
    def __init__(self):
        self.env = rlcard.make('limit-holdem')
        self.state = None
        self.player_id = None
        self.num_actions = self.env.num_actions

    def reset(self):
        self.state, self.player_id = self.env.reset()
        return self.state, self.player_id

    def step(self, action):
        self.state, self.player_id = self.env.step(action)
        return self.state, self.player_id, self.env.is_over()

    def get_payoffs(self):
        return self.env.get_payoffs()

    def get_street(self, public_cards):
        """
        Parameters: public cards/number of cards on the board (list)
        Does: Identifies street status (preflop, flop, turn, river)
        Returns: Integer representation of street (0 - preflop, 3 - flop, 4 - turn, 5 - river)
        """
        if len(public_cards) == 0:
            return 0   # preflop
        elif len(public_cards) == 3:
            return 1   # flop
        elif len(public_cards) == 4:
            return 2   # turn
        else:
            return 3   # river

    def get_hand_strength_bucket(self, hand):
        """
        Parameters: player hand (list)
        Does: Calculates strength of current hand 
        Returns: Integer representation of current hand (0 - weak hand, 1 - mediocre hand, 2 - strong hand)
        """
        ranks = [card[1] for card in hand]

        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, 'T': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }

        values = sorted([rank_map[r] for r in ranks], reverse=True)

        # pair
        if ranks[0] == ranks[1]:
            if values[0] >= 10:
                return 2
            else:
                return 1

        # non-pair
        if values[0] >= 13 and values[1] >= 10:
            return 2
        elif values[0] >= 10:
            return 1
        else:
            return 0
        
    
    

            
    