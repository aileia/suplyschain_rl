import numpy as np
from collections import defaultdict

class LearningCustomer:
    def __init__(self, id, preferred_class, product_classes, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.id = id
        self.preferred_class = preferred_class
        self.product_classes = product_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda: np.zeros(len(product_classes)))
        self.last_state = None
        self.last_action = None

    def get_state(self, prices, availability):
        state = []
        for cls in self.product_classes:
            state.append(prices[cls])
            state.append(1 if availability[cls] else 0)
        return tuple(state)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(self.product_classes))
        else:
            action = np.argmax(self.q_table[state])
        self.last_state = state
        self.last_action = action
        return action

    def update(self, reward, next_state):
        max_q = np.max(self.q_table[next_state])
        current_q = self.q_table[self.last_state][self.last_action]
        self.q_table[self.last_state][self.last_action] += self.alpha * (reward + self.gamma * max_q - current_q)

    def choose_product(self, prices, availability):
        state = self.get_state(prices, availability)
        action_idx = self.select_action(state)
        chosen_product = self.product_classes[action_idx]
        return chosen_product, state