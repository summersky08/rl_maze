if __name__ == '__main__':

	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import random
	
	# create a maze
	maze_size = 15
	num_obstacle = 50
	m = Maze(maze_size, num_obstacle)
	m.describe()

	states = np.reshape(np.arange(maze_size**2), [maze_size,maze_size])
	actions =[[0,1],[-1,0],[0,-1],[1,0]]

	a = Agent(states, actions)

	# learn

	alpha = 0.9
	gamma = 0.9
	epsilon = 0.1
	reward = 10

	Rmat = set_Rmat(states, actions, reward)

	start = states[-1][0]
	goal = states[0][-1]

	state = start

	history = []

	num_step = 100000

	obstacle = m.obstacle

	for step in range(num_step):
	    a.take_action(state, epsilon)
	    next_state = state_transition(state, a.action, maze_size, m.obstacle)
	    a.update_Q(alpha, gamma, Rmat, state, next_state, actions)
	    history.append(state)
	    if next_state == goal:
	        state = start
	        print('step:{} Goal!'.format(step))
	    else:
	        state = next_state
	        
	    if step % (num_step / 100) == 0:
	        counter = [0] * maze_size * maze_size
	        for i in range(len(history)):
	            counter[history[i]] += 1
	        sns.heatmap(np.reshape(np.array(counter), [maze_size, maze_size]))
	        print('step:{} MAP'.format(step))
	        plt.show()

	# visualize Q-value
	sns.heatmap(a.Qmat.max(axis=1).reshape([maze_size, maze_size]))

	#test
	epsilon = 0
	history_test = []

	state = start

	for step in range(100):
	    a.take_action(state, epsilon)
	    next_state = state_transition(state, a.action, maze_size, m.obstacle)
	    if next_state == goal:
	        state = start
	    else:
	        state = next_state
	    history_test.append(state)

	# visualize performance of learned agent
	counter = [0] * maze_size * maze_size
	for i in range(len(history_test)):
	    counter[history_test[i]] += 1
	sns.heatmap(np.reshape(np.array(counter), [maze_size, maze_size]))