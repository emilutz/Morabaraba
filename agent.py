import numpy as np

from state import *

class Agent:

	def __init__(self, name):

		self.name = name

	def make_move(self, current_state):

		next_states = current_state.expand_states()
		if len(next_states) == 0:
			return None
		else:
			return next_states[0]