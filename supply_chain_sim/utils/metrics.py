def compute_metrics(sim_results):
    return {
        "Total Customers": sim_results.get("Total Customers", 0),
        "Fulfilled Orders": sim_results.get("Fulfilled Orders", 0),
        "Fulfillment Rate": sim_results.get("Fulfillment Rate", 0.0),
        "Revenue": sim_results.get("Revenue", 0.0)
    }

def print_metrics(metrics):
    print("\nSimulation Metrics Summary:")
    for key, val in metrics.items():
        print(f"{key}: {val}")