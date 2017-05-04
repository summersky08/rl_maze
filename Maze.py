import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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