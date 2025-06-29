import numpy as np

class RandomProducer:
    def __init__(self, product_classes, price_range=(20, 50)):
        self.product_classes = product_classes
        self.price_range = price_range
        self.price_levels = {cls: 30.0 for cls in product_classes}
        self.queue_state = [0] * len(product_classes)

        # Dummy RL agent
        class DummyAgent:
            def update(self, *args, **kwargs): pass
        self.rl_agent = DummyAgent()

    def observe(self, queue_state):
        self.queue_state = queue_state

    def act(self):
        # Match queue values to product classes
        nonempty_queues = [
            cls for cls, qlen in zip(self.product_classes, self.queue_state)
            if qlen > 0
        ]

        if nonempty_queues:
            product = np.random.choice(nonempty_queues)
        else:
            product = np.random.choice(self.product_classes)

        price = np.random.uniform(*self.price_range)
        self.price_levels[product] = price
        return product, price, None

    def get_prices(self):
        return self.price_levels
