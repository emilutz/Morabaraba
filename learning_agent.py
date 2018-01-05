import math
import numpy as np

from agent import *
from state import *


class LearningAgent(Agent):

	def __init__(self, name, player_index):
		Agent.__init__(self, name)

		self.name = name
		self.player_index = player_index


	def totuple(self, arr):
		"""Turn numpy array into a tuple"""

		try:
		    return tuple(self.totuple(i) for i in arr)
		except TypeError:
		    return arr


	def state_descriptor(self, state, player):
		"""Turn the given state into a more relevant
		descriptor for the training phase"""

		board = np.copy(state.board)

		np.place(board, board == -1, 0)
		np.place(board, board == player, 3)
		np.place(board, board == 3 - player, -1)
		np.place(board, board == 3, 1)

		my_cows = state.cows[player - 1]
		en_cows = state.cows[2 - player]

		return (self.totuple(board), my_cows, en_cows)


	def create_data(self, episodes, agents):
		"""Run a series of episodes between some agents
		and create a dataset for value approximation"""

		state_dict = {}
		alpha = 0.95

		# iterate over episodes
		for e in range(episodes):
			print('Episode {0}/{1}'.format(e + 1, episodes), flush=True)

			# choose 2 random agents to play
			perm = np.random.permutation(len(agents))

			agent1 = agents[perm[0]]
			agent2 = agents[perm[1]]

			agent1.player_index = 1
			agent2.player_index = 2
			players = [agent1, agent2]

			# initialize the state
			current_state = State()
			last_state = None

			# discarding boolean
			discard = False

			# keep track of the states
			states_track_1 = []
			states_track_2 = []

			# final rewards variables
			outcome_1 = 0
			outcome_2 = 0

			# start the episode
			while True:

				# ending condition
				if current_state == None:
					windex = last_state.winner
					# print(self.state_descriptor(last_state, 1))
					# print(self.state_descriptor(last_state, 2))
					# print(windex)

					if windex == 1:
						outcome_1 = 1
						outcome_2 = -1

					if windex == 2:
						outcome_1 = -1
						outcome_2 = 1

					break

				# next move
				else:

					# verify for infinite loops
					if states_track_1.count(self.state_descriptor(current_state, 1)) > 2:
						discard = True
						# print('Loop')
						# print(self.state_descriptor(current_state, 1))
						break

					states_track_1.append(self.state_descriptor(current_state, 1))
					states_track_2.append(self.state_descriptor(current_state, 2))

					to_move = current_state.player_to_move - 1
					last_state = current_state
					current_state = players[to_move].make_move(current_state)

			# get rid of games that ended up in infinite loop
			if discard:
				continue

			# update states from the end to the beginning
			states_track_1.reverse()
			states_track_2.reverse()

			# update the counter and value for each state
			for i in range(len(states_track_1)):

				# update for the first player
				try:
					state_dict[states_track_1[i]][0] += 1
					state_dict[states_track_1[i]][1] += (alpha ** i) * outcome_1
				except KeyError:
					state_dict[states_track_1[i]] = [1, (alpha ** i) * outcome_1]

				# update for the second player
				try:
					state_dict[states_track_2[i]][0] += 1
					state_dict[states_track_2[i]][1] += (alpha ** i) * outcome_2
				except KeyError:
					state_dict[states_track_2[i]] = [1, (alpha ** i) * outcome_2]


		with open('reinforcement_learning_data/states_file.txt', 'w') as f:

			board_data = np.empty((4 * len(state_dict),
									State.BOARD_SIZE,
									State.BOARD_SIZE,
									State.BOARD_SIZE), dtype=np.float32)

			cows_data = np.empty((4 * len(state_dict),
								  2), dtype=np.float32)

			labels = np.empty((4 * len(state_dict),), dtype=np.float32)

			counter = 0
			for key, value in state_dict.items():
				
				f.write('{0} : {1}\n'.format(key, value))

				# rotate board
				for r in range(4):
					board_data[counter] = np.rot(np.asarray(key[0]), k=rot, axis=(1, 2))
					cows_data[counter] = np.asarray([key[1], key[2]])
					labels[counter] = value[1] / value[0]

					counter += 1

			board_data.dump('reinforcement_learning_data/board_data.dat')
			cows_data.dump('reinforcement_learning_data/cows_data.dat')
			labels.dump('reinforcement_learning_data/labels.dat')











	def make_move(self, current_state):
		"""Enforce the agent to make his move"""

		