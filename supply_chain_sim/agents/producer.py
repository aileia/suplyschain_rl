class Producer:
    def __init__(self, product_classes, rl_agent):
        self.product_classes = product_classes
        self.rl_agent = rl_agent
        self.price_levels = {cls: 30.0 for cls in product_classes}
        self.current_product = product_classes[0]

    def observe(self, state):
        self.state = state

    def act(self):
        action_idx = self.rl_agent.select_action(self.state)
        product_idx = action_idx // 3
        price_change = (-5, 0, +5)[action_idx % 3]
        product = self.product_classes[product_idx]
        self.current_product = product
        self.price_levels[product] = max(5, self.price_levels[product] + price_change)
        return product, self.price_levels[product], action_idx

    def get_prices(self):
        return self.price_levels