from agents.producer import Producer
from rl.q_learning import QLearningAgent
from environment.simulator import SupplyChainSim
from utils.metrics import compute_metrics, print_metrics
from visualization.plot_simulation import plot_simulation

product_classes = ["A", "B", "C"]
n_actions = len(product_classes) * 3

rl_agent = QLearningAgent(n_actions)
producer = Producer(product_classes, rl_agent)

sim = SupplyChainSim(producer, product_classes)
results = sim.run(steps=1000)

metrics = compute_metrics(results)
print_metrics(metrics)
plot_simulation(sim)  # 👈 show plots