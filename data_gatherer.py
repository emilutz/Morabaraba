import os
import numpy as np

board_size = 3


if __name__ == '__main__':

	board_data = np.empty((0, board_size, board_size, board_size), dtype=np.float32)
	cows_data = np.empty((0, 2), dtype=np.float32)
	labels = np.empty((0,), dtype=np.float32)

	# iterate over data dumps and collect them
	for i in range(10):

		b_data = np.load(os.path.join('reinforcement_learning_data', 'board_data_' + str(i) + '.dat'))
		c_data = np.load(os.path.join('reinforcement_learning_data', 'cows_data_' + str(i) + '.dat'))
		l_data = np.load(os.path.join('reinforcement_learning_data', 'labels_' + str(i) + '.dat'))

		board_data = np.append(board_data, b_data, axis=0)
		cows_data  = np.append(cows_data, c_data, axis=0)
		labels     = np.append(labels, l_data, axis=0)

	print(board_data.shape)
	print(cows_data.shape)
	print(labels.shape)

	board_data.dump(os.path.join('reinforcement_learning_data_final', 'board_data.dat'))
	cows_data.dump(os.path.join('reinforcement_learning_data_final', 'cows_data.dat'))
	labels.dump(os.path.join('reinforcement_learning_data_final', 'labels.dat'))