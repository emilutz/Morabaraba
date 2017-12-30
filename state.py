import numpy as np
import matplotlib.pyplot as plt
from copy import copy

class State:

	BOARD_SIZE = 3
	GAP_SPOT = 1
	MAX_COWS = 12
	MIN_COWS = 3
	EMPTY = 0

	def __init__(self, board=np.zeros(
		(BOARD_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
	    player_to_move=1, cows=[MAX_COWS, MAX_COWS],
	    can_capture=False, winner=None):

		# make sure this board spot remains invalid
		board[:, self.GAP_SPOT, self.GAP_SPOT] = -1
		self.board = board

		# set the player to move
		self.player_to_move = player_to_move

		# set the number of cows
		self.cows = cows

		# set the flags
		self.can_capture = can_capture

		# the winner variable (only assigned at the end of the game)
		self.winner = winner


	def check_mill(self, player, level, row, column):
		"""Check if a new mill has been formed"""

		# column mill
		if np.all(self.board[level, row, :] == player):
			return True

		# row mill
		if np.all(self.board[level, :, column] == player):
			return True

		# level mill
		if np.all(self.board[:, row, column] == player):
			return True

		return False


	def all_cows_in_mills(self, player):
		"""Check if player has all his cows in mills"""

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):
					if self.board[l, r, c] == player and \
					   not self.check_mill(player, l, r, c):
						return False

		return True


	def move_cow(self, player, source, destination):
		"""Move cow from the source tuple to the destination tuple"""

		# create the new board
		new_board = np.copy(self.board)
		new_board[source] = State.EMPTY
		new_board[destination] = self.player_to_move

		# change the turn
		new_player_to_move = 3 - self.player_to_move

		# create the new state
		new_state = State(new_board,
			              new_player_to_move,
			              copy(self.cows))

		# verify if mill has been formed
		if new_state.check_mill(self.player_to_move,
		 destination[0],
		 destination[1],
		 destination[2]):
			new_state.can_capture = True
			new_state.player_to_move = self.player_to_move									

		return new_state


	def game_over(self):
		"""Checks if one of the players has less than the minimum
		admissible number of cows left and sets the other player
		as the winner of the match; It also declares a draw if both
		players have the minimum amount of cows left"""

		num_cows = [None, None]

		for i in range(2):
			num_cows[i] = self.cows[i] + np.sum(self.board == i + 1)
			if num_cows[i] < State.MIN_COWS:
				self.winner =  2 - i
				return True

		if num_cows == [State.MIN_COWS, State.MIN_COWS]:
			self.winner = 0
			return True

		return False



	def expand_states(self):
		"""Returns a list of possible next states"""

		# game is over
		if self.game_over():
			return []


		states_expanded = []
		opponent = 3 - self.player_to_move

		# capture cow state
		if self.can_capture:
			for l in range(State.BOARD_SIZE):
				for r in range(State.BOARD_SIZE):
					for c in range(State.BOARD_SIZE):

						# opponent cow
						if self.board[l, r, c] == opponent:
							if not self.check_mill(opponent, l, r, c) or \
							   self.all_cows_in_mills(opponent):
								
								# create the new board
								new_board = np.copy(self.board)
								new_board[l, r, c] =State.EMPTY

								# create the new state
								new_state = State(new_board,
									              opponent,
									              copy(self.cows))

								states_expanded.append(new_state)

		# normal state
		else:

			# still cows to be added
			if self.cows[self.player_to_move - 1] > 0:
				new_cows_number = self.cows[self.player_to_move - 1] - 1
				for l in range(State.BOARD_SIZE):
					for r in range(State.BOARD_SIZE):
						for c in range(State.BOARD_SIZE):
							if (r, c) == (State.GAP_SPOT, State.GAP_SPOT):
								continue
							if self.board[l, r, c] == State.EMPTY:

								# create the new board
								new_board = np.copy(self.board)
								new_board[l, r, c] = self.player_to_move

								# change the turn
								new_player_to_move = opponent

								# update the cows
								new_cows = copy(self.cows)
								new_cows[self.player_to_move - 1] -= 1

								# create the new state
								new_state = State(new_board,
									              new_player_to_move,
									              cows=new_cows)

								# verify if mill has been formed
								if new_state.check_mill(self.player_to_move, l, r, c):
									new_state.can_capture = True
									new_state.player_to_move = self.player_to_move	

								states_expanded.append(new_state)

								
			# no more cows left to be added
			else:

				# moving mode
				if np.sum(self.board == self.player_to_move) > 3:
					for l in range(State.BOARD_SIZE):
						for r in range(State.BOARD_SIZE):
							for c in range(State.BOARD_SIZE):
								if self.board[l, r, c] == self.player_to_move:

									# check adjacent spots for availability
									# on level
									if l - 1 >= 0 and self.board[l - 1, r, c] == State.EMPTY:

										new_state = self.move_cow(self.player_to_move,
											(l, r, c), (l - 1, r, c))
										states_expanded.append(new_state)

									if l + 1 < State.BOARD_SIZE and self.board[l + 1, r, c] == State.EMPTY:

										new_state = self.move_cow(self.player_to_move,
											(l, r, c), (l + 1, r, c))
										states_expanded.append(new_state)

									# on row
									if r - 1 >= 0 and self.board[l, r - 1, c] == State.EMPTY:

										new_state = self.move_cow(self.player_to_move,
											(l, r, c), (l, r - 1, c))
										states_expanded.append(new_state)

									if r + 1 < State.BOARD_SIZE and self.board[l, r + 1, c] == State.EMPTY:

										new_state = self.move_cow(self.player_to_move,
											(l, r, c), (l, r + 1, c))
										states_expanded.append(new_state)

									# on column
									if c - 1 >= 0 and self.board[l, r, c - 1] == State.EMPTY:

										new_state = self.move_cow(self.player_to_move,
											(l, r, c), (l, r, c - 1))
										states_expanded.append(new_state)

									if c + 1 < State.BOARD_SIZE and self.board[l, r, c + 1] == State.EMPTY:

										new_state = self.move_cow(self.player_to_move,
											(l, r, c), (l, r, c + 1))
										states_expanded.append(new_state)
				# flying mode
				else:
					for l in range(State.BOARD_SIZE):
						for r in range(State.BOARD_SIZE):
							for c in range(State.BOARD_SIZE):
								if self.board[l, r, c] == self.player_to_move:
									for l2 in range(State.BOARD_SIZE):
										for r2 in range(State.BOARD_SIZE):
											for c2 in range(State.BOARD_SIZE):
												if self.board[l2, r2, c2] == State.EMPTY:
													new_state = self.move_cow(self.player_to_move,
														            (l, r, c), (l2, r2, c2))
													states_expanded.append(new_state)

		# player has nowhere to move and therefore he lost
		if states_expanded == []:
			self.winner = 3 - self.player_to_move

		return states_expanded