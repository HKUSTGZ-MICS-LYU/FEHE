import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import glob

# Read all available accuracy logs
client_accuracy = {}
# 使用 glob 查找所有客户端的日志文件
for csv_path in glob.glob("client_*_accuracy.csv"):
    # 从文件名中提取客户端ID
    client_id = int(csv_path.split('_')[1])
    client_accuracy[client_id] = pd.read_csv(csv_path)

if not client_accuracy:
    print("No accuracy logs found!")
    exit()

# Find the maximum round number across all clients
max_round = 0
for data in client_accuracy.values():
    max_round = max(max_round, data["round"].max())

# Create a complete round sequence
all_rounds = pd.DataFrame({"round": range(1, max_round + 1)})

# Calculate mean accuracy with proper handling of missing values
mean_accuracies = []
for round_num in range(1, max_round + 1):
    round_accuracies = []
    for data in client_accuracy.values():
        if round_num in data["round"].values:
            acc = data[data["round"] == round_num]["accuracy"].iloc[0]
            round_accuracies.append(acc)
    if round_accuracies:  # 如果这一轮有数据
        mean_accuracies.append(np.mean(round_accuracies))
    else:
        mean_accuracies.append(np.nan)  # 使用 NaN 标记缺失值

mean_accuracy = pd.DataFrame({
    "round": range(1, max_round + 1),
    "accuracy": mean_accuracies
})

# Get max mean accuracy and its round (ignoring NaN values)
max_mean_accuracy = mean_accuracy["accuracy"].max()
max_accuracy_round = mean_accuracy.loc[mean_accuracy["accuracy"].idxmax(), "round"]

# Create figure with adjusted size to accommodate legend
plt.figure(figsize=(14, 7))

# 设置背景色
plt.gca().set_facecolor('#f8f9fa')
plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Plot each client's accuracy with better colors
colors = plt.cm.Pastel1(np.linspace(0, 1, len(client_accuracy)))
for (client_id, data), color in zip(client_accuracy.items(), colors):
    plt.plot(data["round"], data["accuracy"], 
            label=f"Client {client_id}", 
            alpha=0.6,
            color=color,
            zorder=2)

# Plot the mean accuracy with enhanced style
plt.plot(mean_accuracy["round"], mean_accuracy["accuracy"], 
         label=f"Mean Accuracy (Max: {max_mean_accuracy:.4f})", 
         color="red", 
         linewidth=3, 
         linestyle="--",
         zorder=3)

# Add marker for maximum mean accuracy
plt.plot(max_accuracy_round, max_mean_accuracy, 'r*', 
         markersize=20,
         zorder=4)

# Add text annotation for max accuracy with enhanced style
plt.annotate(f'Max: {max_mean_accuracy:.4f}\nRound: {int(max_accuracy_round)}',
            xy=(max_accuracy_round, max_mean_accuracy),
            xytext=(20, 20), 
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', 
                     fc='yellow', 
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=2),
            arrowprops=dict(arrowstyle='fancy',
                          fc='0.6',
                          ec='none',
                          connectionstyle='arc3,rad=-0.3'))

# Enhance labels and title
plt.xlabel("Round", fontsize=12, weight='bold')
plt.ylabel("Accuracy", fontsize=12, weight='bold')
plt.title("Client Accuracy Over Time", fontsize=16, weight='bold', pad=20)

# Enhance legend
plt.legend(bbox_to_anchor=(1.05, 1), 
          loc='upper left', 
          borderaxespad=0.,
          frameon=True,
          fancybox=True,
          shadow=True)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# Make tick labels bold
plt.xticks(weight='bold')
plt.yticks(weight='bold')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Save with white background
plt.savefig("client_accuracy.png", 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

# Print the statistics
print(f"\nMean Accuracy Maximum: {max_mean_accuracy:.4f} (Round {int(max_accuracy_round)})")