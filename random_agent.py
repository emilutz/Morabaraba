import numpy as np

from agent import *

class RandomAgent(Agent):
	"""Blind agent that just acts randomly"""

	def __init__(self, name, player_index):
		Agent.__init__(self, name)

		self.name = name
		self.player_index = player_index


	def make_move(self, current_state):

		possible_moves = current_state.expand_states()

		if len(possible_moves) == 0:
			return None

		rnd = np.random.randint(0, len(possible_moves))
		return possible_moves[rnd]