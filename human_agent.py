import numpy as np

from agent import *

class HumanAgent(Agent):

	def __init__(self, name):
		Agent.__init__(self, name)

		self.name = name

	def make_move(self, current_state):

		return current_state