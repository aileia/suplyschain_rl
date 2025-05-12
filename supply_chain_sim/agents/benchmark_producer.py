import numpy as np

class BenchmarkProducer:
    def __init__(self, product_classes, base_price=5.0, price_step=1.0):
        self.product_classes = product_classes
        self.price_levels = {cls: base_price for cls in product_classes}
        self.base_price = base_price
        self.price_step = price_step
        self.current_product = product_classes[0]
        self.state = np.array([])

    def observe(self, queue_state):
        if not isinstance(queue_state, np.ndarray):
            try:
                queue_state = np.array(queue_state)
            except Exception as e:
                print(f"Error converting queue_state to numpy array: {e}")
                self.state = np.array([])
                return
        self.state = queue_state

    def act(self):
        if self.state.size == 0:
            print("Warning: self.state is empty in BenchmarkProducer.act(). Returning default values.")
            product = self.current_product
            price = self.base_price
            self.price_levels[product] = price
            return product, price

        idx = np.argmax(self.state)
        product = self.product_classes[idx]
        self.current_product = product
        price = self.base_price + self.price_step * self.state[idx]
        self.price_levels[product] = price
        return product, price

    def get_prices(self):
        return self.price_levels
