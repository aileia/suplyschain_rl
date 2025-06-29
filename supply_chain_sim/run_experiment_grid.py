import os
import json
from agents.producer import Producer
from agents.benchmark_producer import BenchmarkProducer
from agents.random_producer import RandomProducer
from rl.q_learning import QLearningAgent
from environment.simulator import SupplyChainSim
from utils.metrics import compute_metrics

product_classes = ["A", "B", "C"]
n_actions = len(product_classes) * 3
output_dir = "experiments/metrics"
os.makedirs(output_dir, exist_ok=True)

def create_producer(mode):
    if mode == "q":
        return Producer(product_classes, QLearningAgent(n_actions))
    elif mode == "benchmark":
        return BenchmarkProducer(product_classes)
    elif mode == "random":
        return RandomProducer(product_classes)
    else:
        raise ValueError("Unknown mode")

for arrival_rate in [0.1, 0.5, 0.9]:
    for producer_mode in ["q", "benchmark", "random"]:
        for rep in range(10):
            print(f"Running {producer_mode}, λ={arrival_rate}, rep={rep+1}")
            producer = create_producer(producer_mode)
            sim = SupplyChainSim(producer, product_classes, lambda_arrival=arrival_rate)
            results = sim.run(steps=100_000)
            metrics = compute_metrics(results)
            output_file = os.path.join(
                output_dir, f"{producer_mode}_lambda_{arrival_rate}_run_{rep+1}.json"
            )
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)