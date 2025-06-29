from agents.producer import Producer
from agents.benchmark_producer import BenchmarkProducer
from agents.random_producer import RandomProducer  
from rl.q_learning import QLearningAgent
from environment.simulator import SupplyChainSim
from utils.metrics import compute_metrics, print_metrics
from visualization.plot_simulation import plot_simulation 

def run_experiment(model_name, producer, product_classes, steps=100000, visualize=False):
    print(f"\n--- Running Experiment: {model_name} ---")
    sim = SupplyChainSim(producer, product_classes)
    results = sim.run(steps=steps)
    metrics = compute_metrics(results)
    print_metrics(metrics)
    if visualize:
        plot_simulation(sim)

if __name__ == "__main__":
    product_classes = ["A", "B", "C"]
    n_actions = len(product_classes) * 3

    # Experiment 1: Q-learning Producer
    rl_agent = QLearningAgent(n_actions)
    q_producer = Producer(product_classes, rl_agent)
    run_experiment("Q-learning Producer", q_producer, product_classes, visualize=True)

    # Experiment 2: Benchmark Producer (Greedy Queue Heuristic)
    benchmark_producer = BenchmarkProducer(product_classes)
    run_experiment("Benchmark Producer", benchmark_producer, product_classes, visualize=True)

    #  Experiment 3: Random Producer
    random_producer = RandomProducer(product_classes)
    run_experiment("Random Producer", random_producer, product_classes, visualize=True)
