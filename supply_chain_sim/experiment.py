from agents.producer import Producer
from agents.benchmark_producer import BenchmarkProducer
from rl.q_learning import QLearningAgent
from environment.simulator import SupplyChainSim
from utils.metrics import compute_metrics, print_metrics

def run_experiment(model_name, producer, product_classes, steps=1000):
    sim = SupplyChainSim(producer, product_classes)
    results = sim.run(steps=steps)
    metrics = compute_metrics(results)
    print(f"\n--- Results for {model_name} ---")
    print_metrics(metrics)

if __name__ == "__main__":
    product_classes = ["A", "B", "C"]
    n_actions = len(product_classes) * 3

    # Experiment 1: Q-Learning Agent
    q_agent = QLearningAgent(n_actions)
    q_producer = Producer(product_classes, q_agent)
    run_experiment("Q-Learning Producer", q_producer, product_classes)

    # Experiment 2: Benchmark Heuristic Agent
    benchmark_producer = BenchmarkProducer(product_classes)
    run_experiment("Benchmark Heuristic Producer", benchmark_producer, product_classes)
