
import numpy as np
from collections import deque

class ProductQueue:
    def __init__(self, service_rate=1.5):
        self.queues = {}           # product_class → queue of (customer_id, arrival_time)
        self.service_rate = service_rate
        self.service_in_progress = {}  # product_class → (customer_id, ready_time)

    def enqueue(self, product_class, customer_id, timestamp):
        if product_class not in self.queues:
            self.queues[product_class] = deque()
        self.queues[product_class].append((customer_id, timestamp))

    def start_service(self, product_class, current_time):
        """
        Called when producer selects a product to serve.
        Starts service for one customer if not already in service.
        """
        if product_class in self.queues and product_class not in self.service_in_progress:
            if self.queues[product_class]:
                customer_id, enqueue_time = self.queues[product_class][0]
                service_time = np.random.exponential(1 / self.service_rate)
                ready_time = current_time + int(np.ceil(service_time))
                self.service_in_progress[product_class] = (customer_id, ready_time)

    def dequeue_ready(self, product_class, current_time):
        """
        Checks if the current customer is ready to be served.
        If so, remove from queue and clear in-progress slot.
        """
        if product_class in self.service_in_progress:
            customer_id, ready_time = self.service_in_progress[product_class]
            if current_time >= ready_time:
                self.queues[product_class].popleft()
                del self.service_in_progress[product_class]
                return (customer_id, ready_time)
        return None

    def get_queue_state(self):
        return {cls: len(q) for cls, q in self.queues.items()}
