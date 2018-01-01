import numpy as np

from agent import *
from search_state import *


class SearchAgent(Agent):

	def __init__(self, name, player_index, depth):
		Agent.__init__(self, name)

		self.name = name
		self.player_index = player_index
		self.max_depth = depth


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

		print('Processing node depth', depth)

		# test depth limit condition
		if depth == self.max_depth:
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
			if node.alpha > node.beta:
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

		root = SearchState(current_state.board,
		                   current_state.player_to_move,
		                   current_state.cows,
		                   current_state.can_capture,
		                   current_state.winner)

		self.process_node(root, 0)
		return root.best_child