import numpy as np

class BenchmarkProducer:
    def __init__(self, product_classes, base_price=30.0, price_step=1.0):
        self.product_classes = product_classes
        self.price_levels = {cls: base_price for cls in product_classes}
        self.base_price = base_price
        self.price_step = price_step
        self.current_product = product_classes[0]
        self.state = [0] * len(product_classes)

        class DummyAgent:
            def update(self, *args, **kwargs): pass
        self.rl_agent = DummyAgent()  # Ensures compatibility with simulator

    def observe(self, queue_state):
        self.state = queue_state

    def act(self):
        if sum(self.state) > 0:
            # Queue exists: pick most demanded product
            idx = np.argmax(self.state)
        else:
            # No demand: pick random product
            idx = np.random.randint(len(self.product_classes))

        product = self.product_classes[idx]
        self.current_product = product

        # Adjust price slightly to simulate market dynamics
        noise = np.random.choice([-1, 0, 1]) * self.price_step
        price = max(5, self.price_levels[product] + noise)
        self.price_levels[product] = price

        return product, price, None

    def get_prices(self):
        return self.price_levels
