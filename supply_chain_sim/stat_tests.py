import os
import json
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind

input_dir = "experiments/metrics"
metrics =  ["Fulfillment Rate", "Revenue", "Total Customers"]

# DataFrame collection
records = []

for file in os.listdir(input_dir):
    if not file.endswith(".json"):
        continue
    parts = file.replace(".json", "").split("_")
    producer, lam, run = parts[0], float(parts[2]), int(parts[4])
    with open(os.path.join(input_dir, file)) as f:
        data = json.load(f)
    row = {"producer": producer, "lambda": lam, "run": run}
    row.update(data)
    records.append(row)

df = pd.DataFrame(records)

# Run ANOVA for each λ and metric
for lam in [0.1, 0.5, 0.9]:
    print(f"\n--- Arrival Rate λ = {lam} ---")
    subset = df[df["lambda"] == lam]
    for metric in metrics:
        groups = [group[metric].values for name, group in subset.groupby("producer")]
        fstat, pval = f_oneway(*groups)
        print(f"{metric.title()} ANOVA: F = {fstat:.4f}, p = {pval:.4e}")
        if pval < 0.05:
            print(" → Significant difference between producers.")
        else:
            print(" → No significant difference.")

# Optional: t-test between q and benchmark under λ = 0.9
q = df[(df["lambda"] == 0.9) & (df["producer"] == "q")]["Fulfillment Rate"]
b = df[(df["lambda"] == 0.9) & (df["producer"] == "benchmark")]["Fulfillment Rate"]
tstat, pval = ttest_ind(q, b, equal_var=False)
print(f"\nQ vs Benchmark (λ=0.9) Fulfillment t-test: t = {tstat:.4f}, p = {pval:.4e}")
