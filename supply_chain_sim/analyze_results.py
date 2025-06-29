import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

input_dir = "experiments/metrics"
data = defaultdict(list)

for file in os.listdir(input_dir):
    if file.endswith(".json"):
        parts = file.replace(".json", "").split("_")
        mode = parts[0]
        lam = float(parts[2])
        path = os.path.join(input_dir, file)
        with open(path, "r") as f:
            result = json.load(f)
        data[(mode, lam)].append(result)

# Aggregate and plot
metrics_to_plot = ["Fulfillment Rate", "Revenue", "Total Customers"]
os.makedirs("experiments/plots", exist_ok=True)
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 5))
    for mode in ["q", "benchmark", "random"]:
        x = []
        means = []
        stds = []
        for lam in [0.1, 0.5, 0.9]:
            vals = [d[metric] for d in data[(mode, lam)]]
            x.append(lam)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        plt.errorbar(x, means, yerr=stds, label=mode, capsize=5, marker='o')

    plt.title(f"{metric.replace('_', ' ').title()} vs Arrival Rate")
    plt.xlabel("Arrival Rate (λ)")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"experiments/plots/{metric}_vs_lambda.png")
    plt.close()