import numpy as np

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