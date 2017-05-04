import numpy as np
import random

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