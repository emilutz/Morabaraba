import math
import numpy as np

from agent import *
from search_state import *


class SearchAgent(Agent):

	def __init__(self, name, player_index, expanding_complexity):
		Agent.__init__(self, name)

		self.name = name
		self.player_index = player_index
		self.expanding_complexity = expanding_complexity
		self.depth_limit = 0


	def pass_up(self, node, value, child):
		"""The action of finishing to process a node and
		passing up its results to the parent node"""

		if node == None:
			return

		if node.is_maximizer:

			# update with a better option
			if node.value < value:
				node.value = value
				node.best_child = child

			if node.alpha < value:
				node.alpha = value

		else:

			# update with a better option
			if node.value > value:
				node.value = value
				node.best_child = child

			if node.beta > value:
				node.beta = value


	def process_node(self, node, depth):
		"""The action of processing a node itself"""

		# test depth limit condition
		if depth == self.depth_limit:
			self.pass_up(node.father,
			             node.heuristic_value(self.player_index),
			             node)
			return

		# get the children
		children = node.expand_states()

		if len(children) == 0:
			self.pass_up(node.father,
			             node.heuristic_value(self.player_index),
			             node)
			return

		# iterate through the children and evaluate them but only if it's necessary
		for child in children:

			# test prunning condition
			if node.alpha >= node.beta:
				self.pass_up(node.father, node.value, node)
				return

			# pass parameters to the child
			child.alpha = node.alpha
			child.beta  = node.beta

			self.process_node(child, depth + 1)

		# after expanding all the kids
		self.pass_up(node.father, node.value, node)


	def make_move(self, current_state):
		"""Enforce the agent to make his move"""

		# adjust the depth depending on the branching factor
		new_states = current_state.expand_states()
		if len(new_states) > 0:
			branching_factor = max(len(new_states), 
				                   len(new_states[0].expand_states()))
		else:
			branching_factor = 0

		branching_factor = max(2, branching_factor)

		optimal_depth = math.log(self.expanding_complexity,
		                         branching_factor)

		self.depth_limit = max(1, round(optimal_depth))

		# print('Searching with depth', self.depth_limit)

		# start the tree search
		root = SearchState(current_state.board,
		                   current_state.player_to_move,
		                   current_state.cows,
		                   current_state.can_capture,
		                   current_state.winner)

		self.process_node(root, 0)
		return root.best_child