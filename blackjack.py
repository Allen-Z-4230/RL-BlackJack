import itertools
import random


class Blackjack():
    """Classic Blackjack with a 52 card deck, only modification being that King, Jack, and Queen are treated
    as one card. Player can hit/stand, after standing the dealer takes turns and the game ends.
    The minimum number of cards is two, maximum number is 11
    """
    def __init__(self, mode, verbose=False):
        """Initializes blackjack deck

        mode: one of ['full', 'hidden', 'pomdp']
            full: full observability, dealer's sum is fully visible to player
            hidden: dealer only shows the first card
            pomdp: dealer only shows first card, dealer's 2nd card revealed after termination
        """
        self.mode = mode
        suits = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, and Spades
        values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # 1 ace, 1 of each from 2-9, 4 cards value 10
        self.deck = list(itertools.product(suits, values))
        self.actions = ['hit', 'stick']
        self.terminal = [0,0]

        # game stores player and dealer hands.
        self.ph = []
        self.dh = []

        if verbose:
            print(f'Initilized a {mode} Blackjack game with {len(self.deck)} cards')

    def start(self, ret_hand = False):
        """Start the game by shuffling deck and dealing 2 card to player/dealer (returns s0)
        """
        random.shuffle(self.deck)
        self.ph = [self.deck.pop(), self.deck.pop()]
        self.dh = [self.deck.pop(), self.deck.pop()]
        if ret_hand:
            return self.convert_state((self.ph, self.dh))

    def draw(self):
        return self.deck.pop()

    def evaluate(self, hand):
        """Evaluates a list of cards for the end value

        hand: list of card tuples, the player/dealer's current hand
        """
        cards = [card[1] for card in hand]
        if 'A' in cards:
            a_inds = [i for i, val in enumerate(cards) if val == 'A']
            card_sum = sum([val for val in cards if val != 'A'])
            for a_ind in a_inds:
                if (card_sum + 11 >= 21) and (len(cards) >= 2):  # soft ace
                    card_sum += 1
                else:  # hard ace, if 2 cards or less hard ace will always sum to <21
                    card_sum += 11
        else:
            card_sum = sum(cards)
        return card_sum

    def step_world(self, a):
        """One step in the environment

        s: current state as (player_hand, dealer_hand)
        a: player action
        """
        ph = self.ph
        dh = self.dh
        r = 0

        if a == 'hit':
            ph.append(self.draw())
            s_n = (ph, dh)
            if self.evaluate(ph) > 21:  # player bust
                r = -1
                s_n = [0,0]
        elif a == 'stick':
            while self.evaluate(dh) < 17:
                dh.append(self.draw())
            # play out game outcome
            ps = self.evaluate(ph)
            ds = self.evaluate(dh)
            if (ds > 21) or (ps > ds):
                r = 1
            else:  # tie or lose
                r = -1
            s_n = [0,0]

        return s_n, r

    def step_agent(self, s, a):
        s_n, r = self.step_world(a)
        if s_n == self.terminal:
            if self.mode == 'pomdp':
                return s_n, r, self.evaluate([self.dh[1]]) # return the hidden state
            else:
                return s_n, r, 0
        else:
            s_n_p = self.convert_state(s_n)
            return s_n_p, r, 0 # agent only sees the card sums

    def convert_state(self, s_n):
        if (self.mode == 'pomdp') or (self.mode == 'hidden'):
            s_n_p = [self.evaluate(s_n[0]), self.evaluate([s_n[1][0]])] #only dealer's first card
        elif self.mode == 'full':
            s_n_p = [self.evaluate(s_n[0]), self.evaluate(s_n[1])]
        return s_n_p
