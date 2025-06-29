import numpy as np
from environment.queue import ProductQueue
from agents.customer import LearningCustomer

class SupplyChainSim:
    def __init__(self, producer, product_classes, lambda_arrival=0.5, substitution_threshold=5):
        self.producer = producer
        self.queue = ProductQueue()
        self.product_classes = product_classes
        self.customers = {}
        self.time = 0
        self.lambda_arrival = lambda_arrival
        self.substitution_threshold = substitution_threshold
        self.fulfilled = 0
        self.total_customers = 0
        self.total_revenue = 0
        self.revenue_history = []
        self.fulfillment_history = []
        self.price_history = {cls: [] for cls in product_classes}
        self.queue_history = {cls: [] for cls in product_classes}

        # 🔁 Initialize exponential next arrival
        self.next_arrival_time = self._sample_next_arrival()

    def _sample_next_arrival(self):
        return self.time + int(np.ceil(np.random.exponential(1 / self.lambda_arrival)))


    def step(self):
        prices = self.producer.get_prices()
        availability = {cls: True for cls in self.product_classes}

        if self.time >= self.next_arrival_time:
            customer_id = self.total_customers
            preferred = np.random.choice(self.product_classes)  # random A, B, or C
            customer = LearningCustomer(customer_id, preferred, self.product_classes)
            prices = self.producer.get_prices()
            availability = {cls: True for cls in self.product_classes}
            chosen, state = customer.choose_product(prices, availability)
            self.customers[customer_id] = customer
            self.queue.enqueue(chosen, customer_id, self.time)
            self.total_customers += 1
            self.next_arrival_time = self._sample_next_arrival()  # ⏱️ schedule next arrival

        queue_state = self.queue.get_queue_state()
        state = [queue_state.get(cls, 0) for cls in self.product_classes]
        self.producer.observe(state)
        product, price, action_idx = self.producer.act()

        self.queue.start_service(product, self.time)
        customer_info = self.queue.dequeue_ready(product, self.time)

        reward = price if customer_info else -1
        next_queue_state = self.queue.get_queue_state()
        next_state = [next_queue_state.get(cls, 0) for cls in self.product_classes]
        self.producer.rl_agent.update(reward, next_state)

        if customer_info:
            cust_id = customer_info[0]
            customer = self.customers[cust_id]
            cust_reward = 10
            cust_next_state = customer.get_state(prices, availability)  # ✅ now works
            customer.update(cust_reward, cust_next_state)
            self.fulfilled += 1
            self.total_revenue += price
        # Update time-series tracking for plots
        self.revenue_history.append(self.total_revenue)
        self.fulfillment_history.append(
            self.fulfilled / self.total_customers if self.total_customers else 0
        )

        # Populate price and queue histories
        for cls in self.product_classes:
            self.price_history.setdefault(cls, []).append(self.producer.get_prices()[cls])
            self.queue_history.setdefault(cls, []).append(self.queue.get_queue_state().get(cls, 0))
        self.time += 1




    def run(self, steps=1000):
        for _ in range(steps):
            self.step()
        return {
            "Total Customers": self.total_customers,
            "Fulfilled Orders": self.fulfilled,
            "Fulfillment Rate": self.fulfilled / self.total_customers if self.total_customers else 0,
            "Revenue": self.total_revenue
        }