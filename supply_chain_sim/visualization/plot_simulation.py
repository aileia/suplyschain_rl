import matplotlib.pyplot as plt

def plot_simulation(sim):
    steps = range(len(sim.revenue_history))

    plt.figure(figsize=(14, 10))

    # Revenue
    plt.subplot(2, 2, 1)
    plt.plot(steps, sim.revenue_history, label="Revenue", color="green")
    plt.title("Revenue Over Time")
    plt.xlabel("Step")
    plt.ylabel("Revenue")
    plt.grid(True)

    # Fulfillment Rate
    plt.subplot(2, 2, 2)
    plt.plot(steps, sim.fulfillment_history, label="Fulfillment Rate", color="blue")
    plt.title("Fulfillment Rate Over Time")
    plt.xlabel("Step")
    plt.ylabel("Rate")
    plt.grid(True)

    # Prices
    plt.subplot(2, 2, 3)
    for cls, history in sim.price_history.items():
        plt.plot(steps, history, label=f"Price {cls}")
    plt.title("Dynamic Pricing")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Queue Lengths
    plt.subplot(2, 2, 4)
    for cls, history in sim.queue_history.items():
        plt.plot(steps, history, label=f"Queue {cls}")
    plt.title("Queue Lengths Over Time")
    plt.xlabel("Step")
    plt.ylabel("Queue Size")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
