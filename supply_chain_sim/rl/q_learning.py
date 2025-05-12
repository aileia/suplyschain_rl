import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None

    def select_action(self, state):
        state_key = tuple(state)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(len(self.q_table[state_key]))
        else:
            action = np.argmax(self.q_table[state_key])
        self.last_state = state_key
        self.last_action = action
        return action

    def update(self, reward, next_state):
        next_state_key = tuple(next_state)
        best_next = np.max(self.q_table[next_state_key])
        q = self.q_table[self.last_state][self.last_action]
        self.q_table[self.last_state][self.last_action] += self.alpha * (reward + self.gamma * best_next - q)