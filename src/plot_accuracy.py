import matplotlib.pyplot as plt
import pandas as pd
import os

# Read the accuracy logs
client_accuracy = {}
for client_id in range(10):
    csv_path = f"src/client_{client_id}_accuracy.csv"
    if os.path.exists(csv_path):
        client_accuracy[client_id] = pd.read_csv(csv_path)

# Calculate mean accuracy
mean_accuracy = pd.DataFrame(columns=["round", "accuracy"])
for client_id, data in client_accuracy.items():
    if mean_accuracy.empty:
        mean_accuracy["round"] = data["round"]
        mean_accuracy["accuracy"] = data["accuracy"]
    else:
        mean_accuracy["accuracy"] += data["accuracy"]

mean_accuracy["accuracy"] /= len(client_accuracy)  # Divide by number of clients

# Plot the accuracy
plt.figure(figsize=(10, 6))

# Plot each client's accuracy
for client_id, data in client_accuracy.items():
    plt.plot(data["round"], data["accuracy"], label=f"Client {client_id}", alpha=0.6)

# Plot the mean accuracy
plt.plot(mean_accuracy["round"], mean_accuracy["accuracy"], label="Mean Accuracy", color="black", linewidth=2, linestyle="--")

# Add labels and title
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Client Accuracy Over Time")
plt.legend(loc="lower right")  # Show legend
plt.grid(True)  # Add grid for better readability

# Save the plot
plt.savefig("client_accuracy.png")