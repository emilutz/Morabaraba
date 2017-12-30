import math
import numpy as np

from state import *

class SearchState(State):

	def __init__(self, board=np.zeros(
		(State.BOARD_SIZE, State.BOARD_SIZE, State.BOARD_SIZE),
		dtype=np.int8), player_to_move=1, cows=[State.MAX_COWS, State. MAX_COWS],
	    can_capture=False, winner=None, father=None, is_maximizer=True):

		State.__init__(self, board, player_to_move, cows, can_capture, winner)

		self.father = father
		self.is_maximizer = is_maximizer

		self.alpha = - math.inf
		self.beta  = math.inf

		if is_maximizer: 
			self.value = - math.inf
		else:
			self.value = math.inf


	def expand_states(self):

		next_states = super(SearchState, self).expand_states()
		return [SearchState(s.board, s.player_to_move, s.cows, s.can_capture,
		 s.winner, self, not self.is_maximizer) for s in next_states]


	#-------------------[ feature functions ]-------------------#


	def winning_configuration(self, player):
		"""Returns 1 if hero is the winner, -1 for the villan
		and 0 in all other cases"""

		if self.game_over():
			if self.winner == player:
				return 1
			else:
				return -1

		if len(self.expand_states()) == 0:
			if self.player_to_move == player:
				return -1
			else:
				return 1

		return 0



		if np.sum(self.board == player) + self.cows[player - 1] or \
		   len(expand_states()) == 0:
		   return -1



	def closed_mill(self, player):
		"""Return 1 if the hero closed a mill last round
		and -1 if the villain did so, otherwise return 0"""

		if self.can_capture == False:
			return 0
		elif self.player_to_move == player:
			return 1
		else:
			return -1


	def get_mill_coords(self, player, level, row, column):
		"""Check if a new mill has been formed"""

		# column mill
		if np.all(self.board[level, row, :] == player):
			return [(level, row, c) for c in range(State.BOARD_SIZE)]

		# row mill
		if np.all(self.board[level, :, column] == player):
			return [(level, r, column) for r in range(State.BOARD_SIZE)]

		# level mill
		if np.all(self.board[:, row, column] == player):
			return [(l, row, column) for l in range(State.BOARD_SIZE)]

		return False


	def mills_number(self, player):
		"""Return the difference between hero's number of
		mills and villain's number of mills"""

		hero_mills = []
		villain_mills = []

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):

					mill_cows = self.get_mill_coords(player, l, r, c)
					if mill_cows != False and mill_cows not in hero_mills:
						hero_mills.append(mill_cows)

					mill_cows = self.get_mill_coords(3 - player, l, r, c)
					if mill_cows != False and mill_cows not in villain_mills:
						villain_mills.append(mill_cows)
					
		return len(hero_mills) - len(villain_mills)


	def cow_blocked(self, player, cow_coords):
		"""Checks whether given cow has nowehere to move
		in the moving phase, and returns True if so"""

		if self.board[cow_coords] == player:

			# check level neighbours
			for l in range(cow_coords[0] - 1, cow_coords[0] + 2):
				if l < 0 or l >= State.BOARD_SIZE:
					continue					
				if self.board[l, cow_coords[1], cow_coords[2]] == State.EMPTY:
					return False

			# check row neighbours
			for r in range(cow_coords[1] - 1, cow_coords[1] + 2):
				if r < 0 or r >= State.BOARD_SIZE:
					continue
				if self.board[cow_coords[0], r, cow_coords[2]] == State.EMPTY:
					return False

			# check column neighbours
			for c in range(cow_coords[2] - 1, cow_coords[2] + 2):
				if c < 0 or c >= State.BOARD_SIZE:
					continue
				if self.board[cow_coords[0], cow_coords[1], c] == State.EMPTY:
					return False

			return True

		return False
				
	
	def blocked_cows_number(self, player):
		"""Return the difference between villains's number
		of blocked cows and hero's number of blocked cows"""

		hero_blocked = 0
		villain_blocked = 0

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):

					if self.cow_blocked(player, (l, r, c)):
						hero_blocked += 1

					if self.cow_blocked(3 - player, (l, r, c)):
						villain_blocked += 1

		return villain_blocked - hero_blocked


	def total_cows_difference(self, player):
		"""Return the difference between hero's total number
		of cows and villain's total number of cows"""

		return np.sum(self.board == player) + self.cows[player - 1] - \
			(np.sum(self.board == 3 - player) + self.cows[2 - player])


	def cows_configuration_2(self, player):
		"""Return the difference between hero's number of mills which 
		need one more cow to be completed and villain's"""

		hero_semimills = 0
		villain_semimills = 0

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):

					#-----------------------[ hero ]-----------------------#
					
					# column semi-mill
					if np.sum(self.board[l, r, :] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							hero_semimills += 1

					# row semi-mill
					if np.sum(self.board[l, :, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							hero_semimills += 1

					# level semi-mill
					if np.sum(self.board[:, r, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							hero_semimills += 1

					#-----------------------[ villain ]-----------------------#
					
					# column semi-mill
					if np.sum(self.board[l, r, :] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							villain_semimills += 1

					# row semi-mill
					if np.sum(self.board[l, :, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							villain_semimills += 1

					# level semi-mill
					if np.sum(self.board[:, r, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							villain_semimills += 1

		return hero_semimills - villain_semimills


	def cows_configuration_3(self, player):
		"""Return the difference between hero's number of open-ended semi-mills
		which need one more cow to be completed at either end, and villain's"""

		hero_semimills_count = 0
		villain_semimills_count = 0

		hero_semimills = []
		villain_semimills = []

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):

					#-----------------------[ hero ]-----------------------#
					
					# column semi-mill
					if np.sum(self.board[l, r, :] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							cow_2_confs = ([(l, r, cx) for cx in range(State.BOARD_SIZE)
								if self.board[l, r, cx] != State.EMPTY])
							if len(set(cow_2_confs) & set(hero_semimills)) > 0:
								hero_semimills_count += 1
							hero_semimills += cow_2_confs

					# row semi-mill
					if np.sum(self.board[l, :, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							cow_2_confs = ([(l, rx, c) for rx in range(State.BOARD_SIZE)
								if self.board[l, rx, c] != State.EMPTY])
							if len(set(cow_2_confs) & set(hero_semimills)) > 0:
								hero_semimills_count += 1
							hero_semimills += cow_2_confs

					# level semi-mill
					if np.sum(self.board[:, r, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							cow_2_confs = ([(lx, r, c) for lx in range(State.BOARD_SIZE)
								if self.board[lx, r, c] != State.EMPTY])
							if len(set(cow_2_confs) & set(hero_semimills)) > 0:
								hero_semimills_count += 1
							hero_semimills += cow_2_confs

					#-----------------------[ villain ]-----------------------#
					
					# column semi-mill
					if np.sum(self.board[l, r, :] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							cow_2_confs = ([(l, r, cx) for cx in range(State.BOARD_SIZE)
								if self.board[l, r, cx] != State.EMPTY])
							if len(set(cow_2_confs) & set(villain_semimills)) > 0:
								villain_semimills_count += 1
							villain_semimills += cow_2_confs

					# row semi-mill
					if np.sum(self.board[l, :, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							cow_2_confs = ([(l, rx, c) for rx in range(State.BOARD_SIZE)
								if self.board[l, rx, c] != State.EMPTY])
							if len(set(cow_2_confs) & set(villain_semimills)) > 0:
								villain_semimills_count += 1
							villain_semimills += cow_2_confs

					# level semi-mill
					if np.sum(self.board[:, r, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							cow_2_confs = ([(lx, r, c) for lx in range(State.BOARD_SIZE)
								if self.board[lx, r, c] != State.EMPTY])
							if len(set(cow_2_confs) & set(villain_semimills)) > 0:
								villain_semimills_count += 1
							villain_semimills += cow_2_confs


		print(hero_semimills_count, villain_semimills_count)

		return hero_semimills_count - villain_semimills_count


	def open_mills_2(self, player):
		"""Return the difference between hero's number of mills which 
		need one more cow to be completed in moving phase, and villain's"""

		hero_semimills = 0
		villain_semimills = 0

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):

					#-----------------------[ hero ]-----------------------#
					
					# column semi-mill
					if np.sum(self.board[l, r, :] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[lx, rx, c] == player:
										hero_semimills += 1

					# row semi-mill
					if np.sum(self.board[l, :, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for cx in range(c - 1, c + 2):
									if cx < 0 or cx >= State.BOARD_SIZE:
										continue
									if self.board[lx, r, cx] == player:
										hero_semimills += 1

					# level semi-mill
					if np.sum(self.board[:, r, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for cx in range(c - 1, c + 2):
								if cx < 0 or cx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[l, rx, cx] == player:
										hero_semimills += 1

					#-----------------------[ villain ]-----------------------#
					
					# column semi-mill
					if np.sum(self.board[l, r, :] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[lx, rx, c] == 3 - player:
										villain_semimills += 1

					# row semi-mill
					if np.sum(self.board[l, :, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for cx in range(c - 1, c + 2):
									if cx < 0 or cx >= State.BOARD_SIZE:
										continue
									if self.board[lx, r, cx] == 3 - player:
										villain_semimills += 1

					# level semi-mill
					if np.sum(self.board[:, r, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for cx in range(c - 1, c + 2):
								if cx < 0 or cx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[l, rx, cx] == 3 - player:
										villain_semimills += 1


		return hero_semimills - villain_semimills


	def open_mills_3(self, player):
		"""Return the difference between hero's number of open-ended semi-mills
		which need one more cow to be completed at either end in the moving
		phase, and villain's"""

		hero_semimills_count = 0
		villain_semimills_count = 0

		hero_semimills = []
		villain_semimills = []

		for l in range(State.BOARD_SIZE):
			for r in range(State.BOARD_SIZE):
				for c in range(State.BOARD_SIZE):

					#-----------------------[ hero ]-----------------------#
					
					# column semi-mill
					if np.sum(self.board[l, r, :] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[lx, rx, c] == player:
										cow_2_confs = ([(l, r, cx) for cx in range(State.BOARD_SIZE)
										if self.board[l, r, cx] != State.EMPTY])
										if len(set(cow_2_confs) & set(hero_semimills)) > 0:
											hero_semimills_count += 1
										hero_semimills += cow_2_confs

					# row semi-mill
					if np.sum(self.board[l, :, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for cx in range(c - 1, c + 2):
									if cx < 0 or cx >= State.BOARD_SIZE:
										continue
									if self.board[lx, r, cx] == player:
										cow_2_confs = ([(l, rx, c) for rx in range(State.BOARD_SIZE)
										if self.board[l, rx, c] != State.EMPTY])
										if len(set(cow_2_confs) & set(hero_semimills)) > 0:
											hero_semimills_count += 1
										hero_semimills += cow_2_confs

					# level semi-mill
					if np.sum(self.board[:, r, c] == player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for cx in range(c - 1, c + 2):
								if cx < 0 or cx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[l, rx, cx] == player:
										cow_2_confs = ([(lx, r, c) for lx in range(State.BOARD_SIZE)
										if self.board[lx, r, c] != State.EMPTY])
										if len(set(cow_2_confs) & set(hero_semimills)) > 0:
											hero_semimills_count += 1
										hero_semimills += cow_2_confs

					#-----------------------[ villain ]-----------------------#
					
										# column semi-mill
					if np.sum(self.board[l, r, :] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[lx, rx, c] == 3 - player:
										cow_2_confs = ([(l, r, cx) for cx in range(State.BOARD_SIZE)
										if self.board[l, r, cx] != State.EMPTY])
										if len(set(cow_2_confs) & set(villain_semimills)) > 0:
											villain_semimills_count += 1
										villain_semimills += cow_2_confs

					# row semi-mill
					if np.sum(self.board[l, :, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for lx in range(l - 1, l + 2):
								if lx < 0 or lx >= State.BOARD_SIZE:
									continue
								for cx in range(c - 1, c + 2):
									if cx < 0 or cx >= State.BOARD_SIZE:
										continue
									if self.board[lx, r, cx] == 3 - player:
										cow_2_confs = ([(l, rx, c) for rx in range(State.BOARD_SIZE)
										if self.board[l, rx, c] != State.EMPTY])
										if len(set(cow_2_confs) & set(villain_semimills)) > 0:
											villain_semimills_count += 1
										villain_semimills += cow_2_confs

					# level semi-mill
					if np.sum(self.board[:, r, c] == 3 - player) == 2:
						if self.board[l, r, c] == State.EMPTY:
							for cx in range(c - 1, c + 2):
								if cx < 0 or cx >= State.BOARD_SIZE:
									continue
								for rx in range(r - 1, r + 2):
									if rx < 0 or rx >= State.BOARD_SIZE:
										continue
									if self.board[l, rx, cx] == 3 - player:
										cow_2_confs = ([(lx, r, c) for lx in range(State.BOARD_SIZE)
										if self.board[lx, r, c] != State.EMPTY])
										if len(set(cow_2_confs) & set(villain_semimills)) > 0:
											villain_semimills_count += 1
										villain_semimills += cow_2_confs

		return hero_semimills_count - villain_semimills_count