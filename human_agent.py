import numpy as np

from agent import *

class HumanAgent(Agent):

	def __init__(self, name):
		Agent.__init__(self, name)

		self.name = name