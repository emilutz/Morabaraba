import os
import math
import numpy as np
import tensorflow as tf

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


	def create_data(self, episodes, agents, batch_index):
		"""Run a series of episodes between some agents
		and create a dataset for value approximation"""

		state_dict = {}
		alpha = 0.95

		# iterate over episodes
		for e in range(episodes):
	
			# choose 2 random agents to play
			perm = np.random.permutation(len(agents))

			agent1 = agents[perm[0]]
			agent2 = agents[perm[1]]

			agent1.player_index = 1
			agent2.player_index = 2
			players = [agent1, agent2]

			print('Episode {0}/{1} : {2} vs {3}'.format(
				e + 1, episodes, agent1.name, agent2.name), flush=True)

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


			if (e + 1) % 25 == 0 or e + 1 == episodes:
				print('Saving data...', flush=True)

				with open('reinforcement_learning_data/states_file_' + str(batch_index) + '.txt', 'w') as f:

					augmentations = 6

					board_data = np.empty((augmentations * len(state_dict),
											State.BOARD_SIZE,
											State.BOARD_SIZE,
											State.BOARD_SIZE), dtype=np.float32)

					cows_data = np.empty((augmentations * len(state_dict),
										  2), dtype=np.float32)

					labels = np.empty((augmentations * len(state_dict),), dtype=np.float32)

					counter = 0
					for key, value in state_dict.items():
						
						f.write('{0} : {1}\n'.format(key, value))

						# mirror board
						for fl in range(1, 3):
							board_data[counter] = np.flip(np.asarray(key[0]), axis=fl)
							cows_data[counter] = np.asarray([key[1], key[2]])
							labels[counter] = value[1] / value[0]

							counter += 1

						# rotate board
						for rot in range(4):
							board_data[counter] = np.rot90(np.asarray(key[0]), k=rot, axes=(1, 2))
							cows_data[counter] = np.asarray([key[1], key[2]])
							labels[counter] = value[1] / value[0]

							counter += 1

					board_data.dump('reinforcement_learning_data/board_data_' + str(batch_index) + '.dat')
					cows_data.dump('reinforcement_learning_data/cows_data_' + str(batch_index) + '.dat')
					labels.dump('reinforcement_learning_data/labels_' + str(batch_index) + '.dat')


				# rotate board
				for r in range(4):
					board_data[counter] = np.rot90(np.asarray(key[0]), k=r, axes=(1, 2))
					cows_data[counter] = np.asarray([key[1], key[2]])
					labels[counter] = value[1] / value[0]


	def load_data(self):
		"""Loads the training data from disk"""

		board_data = np.load(os.path.join(
			'reinforcement_learning_data_final',
			'board_data.dat'))
		cows_data = np.load(os.path.join(
			'reinforcement_learning_data_final',
			'cows_data.dat'))
		labels = np.load(os.path.join(
			'reinforcement_learning_data_final',
			'labels.dat'))
		labels = labels.reshape((len(labels), 1))

		permutation = np.random.permutation(len(labels))

		return (board_data[permutation],
		        cows_data[permutation],
		        labels[permutation])




	def train_value_function_approximation(self, make_move=False, child_states=None, children=0):
		"""Train a neural network to do value function approximation"""

		batch_size = 192

		with tf.device('/cpu:0'):

			# data inputs
			board_input = tf.placeholder(tf.float32, name='board_input', shape=[batch_size, 3, 3, 3])
			cows_input  = tf.placeholder(tf.float32, name='cows_input', shape=[batch_size, 2])
			labels      = tf.placeholder(tf.float32, name='labels', shape=[batch_size, 1])

			# dropout probability
			prob = tf.placeholder_with_default(1.0, shape=())


		with tf.device('/gpu:0'):

			with tf.variable_scope('conv', reuse=tf.AUTO_REUSE) as scope:

				kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 3, 64],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))

				bias   = tf.get_variable('bias', initializer=tf.truncated_normal([64],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))

				conv = tf.nn.conv2d(board_input, kernel, [1, 1, 1, 1], padding='VALID')
				out  = tf.nn.bias_add(conv, bias)
				conv_activated = tf.nn.relu(out, name='conv_output')


			with tf.variable_scope('layer_1', reuse=tf.AUTO_REUSE) as scope:

				conv_board_shape = conv_activated.get_shape()
				flat_board_shape = int(np.prod(conv_board_shape[1:]))
				flat_board = tf.reshape(conv_activated, [batch_size, flat_board_shape])

				weights = tf.get_variable('weights', initializer=tf.truncated_normal([flat_board_shape, 128],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))

				bias = tf.get_variable('bias', initializer=tf.truncated_normal([128],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))

				out = tf.nn.bias_add(tf.matmul(flat_board, weights), bias)
				flat_activated = tf.nn.relu(out, name='flat_output')


			with tf.variable_scope('layer_2', reuse=tf.AUTO_REUSE) as scope:

				concatenated = tf.concat([flat_activated, cows_input], axis=1)

				weights = tf.get_variable('weights', initializer=tf.truncated_normal([130, 128],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))

				bias = tf.get_variable('bias', initializer=tf.truncated_normal([128],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))				

				out = tf.nn.bias_add(tf.matmul(concatenated, weights), bias)
				conc_activated = tf.nn.relu(out, name='flat_output')


			with tf.variable_scope('layer_3', reuse=tf.AUTO_REUSE) as scope:

				weights = tf.get_variable('weights', initializer=tf.truncated_normal([128, 1],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))

				bias = tf.get_variable('bias', initializer=tf.truncated_normal([1],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))				

				out = tf.nn.bias_add(tf.matmul(conc_activated, weights), bias)
				output_value = tf.nn.tanh(out, name='output_value')


			with tf.variable_scope('last', reuse=tf.AUTO_REUSE) as scope:

				loss = tf.losses.mean_squared_error(labels, output_value)

				optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-3)
				train_op = optimizer.minimize(loss)


		saver = tf.train.Saver()

		with tf.Session() as sess:

			if make_move:
				saver.restore(sess, "./reinforcement_learning_model/model.ckpt")
				
				feed_dict = {
								board_input : child_states[0],
								cows_input : child_states[1],
								labels : child_states[2]
							}

				rez = sess.run(output_value, feed_dict=feed_dict)[0]
				return np.argmax(rez[:children])

			else:

				# collect data for Tensorboard
				with tf.device('/cpu:0'):

					tf.summary.scalar('loss', loss)
					merged = tf.summary.merge_all()
					tensorboard_writer = tf.summary.FileWriter('tensorboard_data', sess.graph)


				init_op = tf.global_variables_initializer()
				sess.run(init_op)

				input_data = self.load_data()
				dataset_divider = int(5 * len(input_data[0]) // 6)

				training_board = input_data[0][:dataset_divider]
				training_cows = input_data[1][:dataset_divider]
				training_labels = input_data[2][:dataset_divider]

				validation_board = input_data[0][dataset_divider:]
				validation_cows = input_data[1][dataset_divider:]
				validation_labels = input_data[2][dataset_divider:]

				best_loss = 9999

				epoch = 2
				max_epochs = 10

				while epoch < max_epochs:

					# training
					for step in range(0, dataset_divider, batch_size):

						try:
							feed_dict = {
								prob : 0.5,
								board_input : training_board[step:step + batch_size],
								cows_input : training_cows[step:step + batch_size],
								labels : training_labels[step:step + batch_size]
							}

							_, loss_value, summary = sess.run([train_op, loss, merged],
							feed_dict = feed_dict)

							tensorboard_writer.add_summary(summary,
							 (epoch * dataset_divider + step) // batch_size)

						except ValueError:
							pass


					total_loss = 0
					loss_count = 0

					# validation
					for step in range(0, (len(input_data[0]) - dataset_divider) // 6, batch_size):

						try:
							feed_dict = {
								board_input : validation_board[step:step + batch_size],
								cows_input : validation_cows[step:step + batch_size],
								labels : validation_labels[step:step + batch_size]
							}

							loss_value = sess.run(loss, feed_dict = feed_dict)

							total_loss += loss_value
							loss_count += 1

						except ValueError:
							pass

					epoch += 1
					validation_loss = total_loss / loss_count

					if validation_loss < best_loss:
						best_loss = validation_loss
						saver.save(sess, "./reinforcement_learning_model/model.ckpt")

					print('epoch {0} loss {1:.4f}'.format(epoch, validation_loss))



	def make_move(self, current_state):
		"""Enforce the agent to make his move"""

		batch_size = 192

		test_board  = np.zeros((batch_size, 3, 3, 3))
		test_cows   = np.zeros((batch_size, 2))
		test_labels = np.zeros((batch_size, 1)) 

		new_states = current_state.expand_states()

		if len(new_states) == 0:
			return None

		for i, state in enumerate(new_states):

			desc = self.state_descriptor(state, self.player_index)
			test_board[i] = np.asarray(desc[0])
			test_cows[i]  = np.asarray([desc[1], desc[2]])

		return new_states[self.train_value_function_approximation(
			True, (test_board, test_cows, test_labels), len(new_states))]
 

		