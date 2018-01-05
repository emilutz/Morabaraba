import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, RegularPolygon, Shadow

from state import *
from search_state import *

from agent import *
from human_agent import *
from random_agent import *
from search_agent import *
from learning_agent import *



class Main:
	"""The central part of the program"""

	wizard_color = '#F7B738'
	board_color = '#BCE0DF'
	selected_color = '#AFFFD4'
	spot_color = ['#FFFFFF', '#FF0000', '#0000FF']

	fig_size = 5.5
	fig_pad  = 0.25
	spot_radius = 0.15

	fig_mid  = fig_size / 2 
	fig_unit = (fig_mid - fig_pad) / State.BOARD_SIZE


	def __init__(self, agent1, agent2):
		
		# assign the agents
		self.players = [agent1, agent2]

		# start with an empty state
		self.current_state = State()
		self.img_nr = -1

		# create the figure
		self.fig = plt.figure(figsize=(Main.fig_size, Main.fig_size), facecolor=self.board_color)
		self.fig.suptitle('Good Luck !', fontsize=12, fontweight='bold')

		self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9),
                                    aspect='equal', frameon=False,
                                    xlim=(-0.05, Main.fig_size + 0.05),
                                    ylim=(-0.05, Main.fig_size + 0.05))

		self.cows_left = [
		self.ax.text(4 * Main.fig_pad, Main.fig_size, str(self.current_state.cows[0]),
        bbox={'facecolor':Main.spot_color[1], 'alpha':0.5, 'pad':10}),
        self.ax.text(Main.fig_size - 5 * Main.fig_pad, Main.fig_size, str(self.current_state.cows[0]),
        bbox={'facecolor':Main.spot_color[2], 'alpha':0.5, 'pad':10})
        ]

		self.player_names = [
		self.ax.text(-0.5 * Main.fig_pad, Main.fig_size, agent1.name,
		 fontsize=14, color=Main.spot_color[1]),
		self.ax.text(Main.fig_size - 2.5 * Main.fig_pad, Main.fig_size, agent2.name,
		 fontsize=14, color=Main.spot_color[2])]


		for axis in (self.ax.xaxis, self.ax.yaxis):
			axis.set_major_formatter(plt.NullFormatter())
			axis.set_major_locator(plt.NullLocator())

		# create the grid of spots
		self.spots = np.array([[[Circle(
			self.graphic_location(l, r, c), linewidth=2,
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

		# store the selected entity (for moving pieces)
		self.selected = None

		# create event hook for mouse clicks
		self.fig.canvas.mpl_connect('button_press_event', self.button_press)
		self.fig.canvas.mpl_connect('button_release_event', self.button_release)
		

	def graphic_location(self, l, r, c):
		"""Maps the mathematical 3 dimensional index of a board state
		to the location in the graphical user interface board"""

		return (Main.fig_mid + (c - 1) * (State.BOARD_SIZE - l) * Main.fig_unit,
			Main.fig_mid + (r - 1) * (State.BOARD_SIZE - l) * Main.fig_unit)


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


	def spot_clicked(self, x, y):
		"""Maps click locations to actual entities lying on the board"""

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):

					sx, sy = self.graphic_location(l, r, c)

					if math.sqrt((x - sx)**2 + (y - sy)**2) < self.spot_radius:
						if (r, c) == (State.GAP_SPOT, State.GAP_SPOT):
							return "wizard"
						else:
							return (l, r, c)

	def button_press(self, event):

		x, y = map(float, (event.xdata, event.ydata))
		self.selected = self.spot_clicked(x, y)

		try:
			if self.current_state.board[self.selected] == self.current_state.player_to_move \
			and sum(self.current_state.cows) == 0:
				self.spots[self.selected].set_edgecolor(self.selected_color)
				self.fig.canvas.draw()
		except IndexError:
			pass
		except ValueError:
			pass
		

					
	def button_release(self, event):
		"""Handle mouse clicks"""

		player = self.players[self.current_state.player_to_move - 1]

		x, y = map(float, (event.xdata, event.ydata))
		clicked = self.spot_clicked(x, y)

		self.fig.suptitle('')
		self.fig.canvas.draw()


		#------------------------[ player is human ]------------------------#
		
		if isinstance(player, HumanAgent):
			
			# possible next board configurations
			possible_next_states = self.current_state.expand_states()
			possible_configurations = [state.board for state in possible_next_states]

			current_board = np.copy(self.current_state.board)

			# no moves can be made
			if len(possible_next_states) == 0:
				windex = self.current_state.winner
				if windex:
					self.fig.suptitle(self.players[windex - 1].name + ' wins !', fontsize=12, fontweight='bold')
				else:
					self.fig.suptitle('Draw', fontsize=12, fontweight='bold')
				self.fig.canvas.draw()
				return

			# air clicks do not count
			if self.selected == None or clicked == None:
				return

			# not allowed to touch its majesty
			if self.selected == 'wizard' or clicked == 'wizard':
				self.fig.suptitle("Don't touch the wizard !", fontsize=12, fontweight='bold')
				self.fig.canvas.draw()
				return

			# clicked and released same entity
			if clicked == self.selected:
				# add a new cow on an empty spot
				if current_board[clicked] == State.EMPTY:
					current_board[clicked] = self.current_state.player_to_move
				# capture opponent's piece
				elif current_board[clicked] == 3 - self.current_state.player_to_move:
					current_board[clicked] = State.EMPTY
			# clicked on different spots
			elif current_board[self.selected] == self.current_state.player_to_move:
				current_board[self.selected] = State.EMPTY
				current_board[clicked] = self.current_state.player_to_move


			# verify if the move is valid
			state_valid = False
			for i, possible_board in enumerate(possible_configurations):
				if np.all(possible_board == current_board):
					state_valid = True
					state_index = i
					break

			if state_valid:
				self.current_state = possible_next_states[state_index]
				self.update_graphical_board()
			else:
				self.fig.suptitle('Invalid move !', fontsize=12, fontweight='bold')
				self.fig.canvas.draw()


			# flush the mouse press
			if self.selected not in [None, 'wizard']:
				self.spots[self.selected].set_edgecolor(None)
				self.selected = None


		#------------------------[ player is AI ]------------------------#
		
		else:
			if clicked == "wizard" and self.selected == 'wizard':
				next_state = player.make_move(self.current_state)

				# no move returned
				if next_state == None:
					windex = self.current_state.winner
					if windex:
						self.fig.suptitle(self.players[windex - 1].name + ' wins !', fontsize=12, fontweight='bold')
					else:
						self.fig.suptitle('Draw', fontsize=12, fontweight='bold')
					self.fig.canvas.draw()
					return
				else:
					self.current_state = next_state
					self.update_graphical_board()


		if self.current_state.can_capture:
			self.fig.suptitle('Mill ! Shoot a cow !', fontsize=12, fontweight='bold')

		for i in range(2):
			self.cows_left[i].set_text(str(self.current_state.cows[i]))
		
		self.fig.canvas.draw()


	###################### RANDOMELI ######################

	# def display_states_test(self, event):

	# 	board = np.asarray([
	# 		[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
	# 		[[1, 1, 1], [0, -1, 0], [0, 0, 0]],
	# 		[[2, 2, 2], [0, -1, 0], [0, 0, 0]]
	# 		], dtype=np.int8)

	# 	self.current_state = SearchState(board)
	# 	self.current_state.cows = [5, 5]
	# 	self.current_state.player_to_move = 1
	# 	self.current_state.can_capture = False
	# 	self.update_graphical_board()

	# 	print('Initial : to_move ', self.current_state.player_to_move)
	# 	print('          cows    ', self.current_state.cows)
	# 	print('          capture ', self.current_state.can_capture, '\n')


	# 	if self.img_nr == -1:
	# 		self.img_nr = 0
	# 		return

	# 	next_states = self.current_state.expand_states()
	# 	print(self.img_nr, " / ", len(next_states), " states")

	# 	if (len(next_states) == 0):
	# 		print('Winner is ', self.current_state.winner)
	# 		sys.exit()
		
	# 	if self.img_nr < len(next_states):
	# 		self.current_state = next_states[self.img_nr]
	# 		self.update_graphical_board()
	# 		print('After   : to_move ', self.current_state.player_to_move)
	# 		print('          cows    ', self.current_state.cows)
	# 		print('          capture ', self.current_state.can_capture)
	# 		print('        : past cows', self.current_state.father.cows, '\n')
	# 	else:
	# 		sys.exit('no more states')
	# 	self.img_nr += 1


	# def display_features_test(self, event):

	# 	board = np.asarray([
	# 		[[0, 0, 0], [1, -1, 1], [1, 1, 2]],
	# 		[[1, 1, 0], [2, -1, 1], [2, 2, 2]],
	# 		[[1, 2, 1], [1, -1, 0], [2, 1, 1]]
	# 		], dtype=np.int8)

		# board = np.asarray([
		# 	[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
		# 	[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
		# 	[[0, 0, 0], [0, -1, 0], [0, 0, 0]]
		# 	], dtype=np.int8)

		# board[0, 0, 1] = 1

		# self.current_state = SearchState(board)
		# self.current_state.cows = [0, 0]
		# self.current_state.player_to_move = 2
		# self.current_state.can_capture = False
		# self.update_graphical_board()

		# print('closed mill',
		#  self.current_state.closed_mill(self.current_state.player_to_move),
		#  self.current_state.closed_mill(3 - self.current_state.player_to_move))

		# print('mills number',
		#  self.current_state.mills_number(self.current_state.player_to_move),
		#  self.current_state.mills_number(3 - self.current_state.player_to_move))

		# print('blocked cows',
		#  self.current_state.blocked_cows_number(self.current_state.player_to_move),
		#  self.current_state.blocked_cows_number(3 - self.current_state.player_to_move))

		# print('number of cows',
		#  self.current_state.total_cows_difference(self.current_state.player_to_move),
		#  self.current_state.total_cows_difference(3 - self.current_state.player_to_move))

		# print('2 cow confs',
		#  self.current_state.cows_configuration_2(self.current_state.player_to_move),
		#  self.current_state.cows_configuration_2(3 - self.current_state.player_to_move))

		# print('3 cow confs',
		#  self.current_state.cows_configuration_3(self.current_state.player_to_move),
		#  self.current_state.cows_configuration_3(3 - self.current_state.player_to_move))

		# print('2 open mills',
		#  self.current_state.open_mills_2(self.current_state.player_to_move),
		#  self.current_state.open_mills_2(3 - self.current_state.player_to_move))

		# print('3 open mills',
		#  self.current_state.open_mills_3(self.current_state.player_to_move),
		#  self.current_state.open_mills_3(3 - self.current_state.player_to_move))

		# print('winning conf',
		#  self.current_state.winning_configuration(self.current_state.player_to_move),
		#  self.current_state.winning_configuration(3 - self.current_state.player_to_move))

		# print('heuristic value', self.current_state.heuristic_value(1))
		


	# def random_test_sequence(self, event):

	# 	print('State   : to_move ', self.current_state.player_to_move)
	# 	print('          cows    ', self.current_state.cows)
	# 	print('          capture ', self.current_state.can_capture, '\n')
	# 	self.update_graphical_board()

	# 	next_states = self.current_state.expand_states()
	# 	print(len(next_states), ' possible next states')

	# 	if (len(next_states) == 0):
	# 		print('Winner is ', self.current_state.winner)
	# 		sys.exit()

	# 	rnd = np.random.randint(0, len(next_states))
	# 	self.current_state = next_states[rnd]

	#######################################################


if __name__ == '__main__':

	# random = RandomAgent(name='Random', player_index=2)
	# pulica = SearchAgent(name='pulica', player_index=1, expanding_complexity=5000)
	# search = SearchAgent(name='Sotirios', player_index=2, expanding_complexity=2000)
	# human = HumanAgent(name='Dawid')

	# runner = Main(human, search)
	# plt.show()

	rnd = RandomAgent(name='rnd', player_index=1)
	ag1 = SearchAgent(name='ag1', player_index=1, expanding_complexity=200)
	ag2 = SearchAgent(name='ag2', player_index=1, expanding_complexity=400)
	ag3 = SearchAgent(name='ag2', player_index=1, expanding_complexity=600)

	monty = LearningAgent(name='monty', player_index=0)

	monty.create_data(8, [rnd, ag1, ag2, ag3])


	# runner = Main(ag1, ag2)
	# plt.show()