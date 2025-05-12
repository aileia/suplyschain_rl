from collections import deque

class ProductQueue:
    def __init__(self, service_time=20):
        self.queues = {}
        self.service_time = service_time

    def enqueue(self, product_class, customer_id, timestamp):
        if product_class not in self.queues:
            self.queues[product_class] = deque()
        self.queues[product_class].append((customer_id, timestamp))

    def dequeue(self, product_class, current_time):
        if self.queues.get(product_class):
            customer_id, enqueue_time = self.queues[product_class][0]
            if current_time - enqueue_time <= self.service_time:
                return self.queues[product_class].popleft()
        return None

    def get_queue_state(self):
        return {cls: len(q) for cls, q in self.queues.items()}