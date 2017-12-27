import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, RegularPolygon, Shadow

from state import *
from agent import *

class Main:
	"""The central part of the program"""

	wizard_color = '#FCFF54'
	board_color = '#EF924F'
	spot_color = ['#FFFFFF', '#FF0000', '#0000FF']

	fig_size = 5
	fig_pad  = 0.25
	spot_radius = 0.15

	fig_mid  = fig_size / 2 
	fig_unit = (fig_mid - fig_pad) / State.BOARD_SIZE


	def __init__(self):
		
		# start with an empty state
		self.current_state = State()
		self.img_nr = -1

		# create the figure
		self.fig = plt.figure(figsize=(Main.fig_size, Main.fig_size), facecolor=self.board_color)

		self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9),
                                    aspect='equal', frameon=False,
                                    xlim=(-0.05, Main.fig_size + 0.05),
                                    ylim=(-0.05, Main.fig_size + 0.05))

		for axis in (self.ax.xaxis, self.ax.yaxis):
			axis.set_major_formatter(plt.NullFormatter())
			axis.set_major_locator(plt.NullLocator())

		# create the grid of spots
		self.spots = np.array([[[Circle(
			self.graphic_location(l, r, c),
			facecolor=self.spot_color[0], radius=self.spot_radius)
                          for c in range(State.BOARD_SIZE)]
                         for r in range(State.BOARD_SIZE)]
                        for l in range(State.BOARD_SIZE)])


		# draw the connections between spots
		for l in range(State.BOARD_SIZE):
			self.ax.add_patch(ConnectionPatch(
				self.graphic_location(l, 0, 0), self.graphic_location(l, 0, 2),
				"data", "data"))
			self.ax.add_patch(ConnectionPatch(
				self.graphic_location(l, 0, 0), self.graphic_location(l, 2, 0),
				"data", "data"))
			self.ax.add_patch(ConnectionPatch(
				self.graphic_location(l, 0, 2), self.graphic_location(l, 2, 2),
				"data", "data"))
			self.ax.add_patch(ConnectionPatch(
				self.graphic_location(l, 2, 0), self.graphic_location(l, 2, 2),
				"data", "data"))

		for r in range(State.BOARD_SIZE):
			for c in range(State.BOARD_SIZE):
				if (r, c) != (State.GAP_SPOT, State.GAP_SPOT):
					self.ax.add_patch(ConnectionPatch(
				self.graphic_location(0, r, c), self.graphic_location(2, r, c),
				"data", "data", facecolor=self.board_color))

		# draw the spots
		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):
					if (r, c) != (State.GAP_SPOT, State.GAP_SPOT):
						self.ax.add_patch(self.spots[l, r, c])

		# draw the central player turn deciding wizard
		self.wizard = RegularPolygon(
			self.graphic_location(0, State.GAP_SPOT, State.GAP_SPOT),
			numVertices=8, radius=0.2, facecolor=self.wizard_color,
			linewidth=2, linestyle='-', hatch='*',
			edgecolor=self.spot_color[self.current_state.player_to_move])
		self.ax.add_patch(self.wizard)

		# create event hook for mouse clicks
		self.fig.canvas.mpl_connect('button_press_event', self.random_test_sequence)
		

	def graphic_location(self, l, r, c):
		"""Maps the mathematical 3 dimensional index of a board state
		to the location in the graphical user interface board"""

		return (Main.fig_mid + (r - 1) * (State.BOARD_SIZE - l) * Main.fig_unit,
			Main.fig_mid + (c - 1) * (State.BOARD_SIZE - l) * Main.fig_unit)


	def update_graphical_board(self):
		"""Update the graphical board with the context of the current state"""

		# update spots
		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):
					self.spots[l, r, c].set_facecolor(self.spot_color[
						self.current_state.board[l, r, c]])

		# update wizard
		self.wizard.set_edgecolor(self.spot_color[
			self.current_state.player_to_move])
				
		self.fig.canvas.draw()


	def button_press(self, event):
		"""Handle mouse clicks"""

		# x, y = map(float, (event.xdata, event.ydata))

		# for l in range(State.BOARD_SIZE):
		# 	for r in range(State.BOARD_SIZE):
		# 		for c in range(State.BOARD_SIZE):
		# 			if (r, c) != (State.GAP_SPOT, State.GAP_SPOT):
						
		# 				sx, sy = self.graphic_location(l, r, c)

		# 				if math.sqrt((x - sx)**2 + (y - sy)**2) < self.spot_radius:
		# 					self.current_state.board[l, r, c] = self.current_state.player_to_move
		# 					self.current_state.change_turn()
		# 					self.update_graphical_board()
		# 					return



	###################### RANDOMELI ######################

	def display_states_test(self, event):

		board = np.asarray([
			[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
			[[1, 1, 1], [0, -1, 0], [0, 0, 0]],
			[[2, 2, 2], [0, -1, 0], [0, 0, 0]]
			], dtype=np.int8)

		self.current_state = State(board)
		self.current_state.cows = [0, 0]
		self.current_state.player_to_move = 2
		self.current_state.can_capture = False
		self.update_graphical_board()

		print('Initial : to_move ', self.current_state.player_to_move)
		print('          cows    ', self.current_state.cows)
		print('          capture ', self.current_state.can_capture, '\n')


		if self.img_nr == -1:
			self.img_nr = 0
			return

		next_states = self.current_state.expand_states()
		print(self.img_nr, " / ", len(next_states), " states")

		if (len(next_states) == 0):
			print('Winner is ', self.current_state.winner)
			sys.exit()
		
		if self.img_nr < len(next_states):
			self.current_state = next_states[self.img_nr]
			self.update_graphical_board()
			print('After   : to_move ', self.current_state.player_to_move)
			print('          cows    ', self.current_state.cows)
			print('          capture ', self.current_state.can_capture, '\n')
		else:
			sys.exit('no more states')
		self.img_nr += 1


	def random_test_sequence(self, event):

		print('State   : to_move ', self.current_state.player_to_move)
		print('          cows    ', self.current_state.cows)
		print('          capture ', self.current_state.can_capture, '\n')
		self.update_graphical_board()

		next_states = self.current_state.expand_states()
		print(len(next_states), ' possible next states')

		if (len(next_states) == 0):
			print('Winner is ', self.current_state.winner)
			sys.exit()

		rnd = np.random.randint(0, len(next_states))
		self.current_state = next_states[rnd]

	#######################################################

if __name__ == '__main__':

	runner = Main()
	plt.show()