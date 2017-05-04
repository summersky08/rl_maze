import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

class Maze():
    def __init__(self, maze_size=10, num_obst=20):
        obstacle = np.random.choice(maze_size*maze_size - 2, num_obst, replace=False)
         # to avoid to put obstacles in the entrance (lower-left corner) or the exit (upper-right corner)
        for i in range(len(obstacle)):
            if obstacle[i] >= maze_size-1:
                obstacle[i] += 1
                if obstacle[i] >= maze_size*(maze_size-1):
                    obstacle[i] += 1
                    if obstacle[i] >= maze_size*maze_size:
                        obstacle[i] -= maze_size*maze_size
        self.obstacle = list(obstacle)
        self.size = maze_size
    
    def coordinate2index(self, location):
        if type(location) == list and type(location[0]) == list:
            for i in range(len(location)):
                location[i] = np.reshape(np.arange(self.size*self.size), [self.size, self.size])[location[i][0]][location[i][1]]
        elif type(location) == list and len(location) == 2 and type(location[0]) == int:
            location = np.reshape(np.arange(self.size*self.size), [self.size, self.size])[location[0]][location[1]]
        return location
    
    def describe(self):
        self.obstacle = self.coordinate2index(self.obstacle)
    
        self.map_ = np.ones([self.size, self.size])
        self.map_[-1,0], self.map_[0,-1] = 0,0

        self.map_ = np.reshape(self.map_, [1, self.map_.size])

        if type(self.obstacle) == list:
            for i in range(len(self.obstacle)):
                self.map_[0][self.obstacle[i]] = 2

        self.map_[0][self.obstacle] = 2
        self.map_ = np.reshape(self.map_, [self.size, self.size])
        sns.heatmap(self.map_)
        plt.show()


class Agent():
    def __init__(self, states, actions):
        self.Qmat = np.random.random([states.size, len(actions)])
    
    def take_action(self, state, epsilon):
        if random.random() >  epsilon:
            action_idx = self.Qmat[state].argmax()
            self.action = actions[action_idx]
        else:
            self.action = random.choice(actions)
    
    def update_Q(self, alpha, gamma, Rmat, state, next_state, actions):
        self.Qmat[state, actions.index(self.action)] = self.Qmat[state, actions.index(self.action)] + alpha*(Rmat[next_state].max() + gamma*self.Qmat[next_state].max() - self.Qmat[state, actions.index(self.action)])


def coordinate2index(maze_size, location):
    if type(location) == list and type(location[0]) == list:
        for i in range(len(location)):
            location[i] = np.reshape(np.arange(maze_size*maze_size), [maze_size, maze_size])[location[i][0]][location[i][1]]
    elif type(location) == list and len(location) == 2 and type(location[0]) == int:
        location = np.reshape(np.arange(maze_size*maze_size), [maze_size, maze_size])[location[0]][location[1]]
    return location


def state_transition(state, action, maze_size, obstacle):
    obstacle = coordinate2index(maze_size, obstacle)
    
    location = np.where(states == state)
    next_state_index = list([np.array(location)[0] + action[0], np.array(location)[1] + action[1]])
    if max(next_state_index) == maze_size or min(next_state_index) == -1: # cannot go because of wall
        next_state = state
    else:
        next_state = list(states[next_state_index])[0]
    
    if type(obstacle) == int:
        obstacle = [obstacle]
    
    if next_state in obstacle:   # cannot go because of obstacle
        next_state = state
    return next_state


def set_Rmat(states, actions, reward):
    #Rmat = np.zeros([states.size, len(actions)])
    Rmat = np.zeros([states.size, len(actions)]) - reward / 100
    Rmat[maze_size-1] = reward
    return Rmat


def main():
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


if __name__ == '__main__':
	main()