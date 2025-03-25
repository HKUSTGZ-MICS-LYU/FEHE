import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import glob
import re

# Read all available accuracy logs
client_accuracy = {}
# 使用 glob 查找所有客户端的日志文件
for csv_path in glob.glob("/hpc2hdd/home/xmeng027/Work/FEHE/src/Experiment/ResNet18_CIFAR10/50_20/NONIID/client_*_accuracy.csv"):
    # 从文件名中提取客户端ID，使用更可靠的方法
    filename = os.path.basename(csv_path)
    # 使用正则表达式匹配"client_X_accuracy.csv"中的X
    match = re.search(r'client_(\d+)_accuracy\.csv', filename)
    if match:
        client_id = int(match.group(1))
        print(f"找到客户端 {client_id} 的准确率日志: {filename}")
        client_accuracy[client_id] = pd.read_csv(csv_path)
    else:
        print(f"警告: 无法从文件名 {filename} 中提取客户端ID")

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

# 设置全局字体大小
plt.rcParams.update({'font.size': 14})

# Plot each client's accuracy with better colors
colors = plt.cm.Pastel1(np.linspace(0, 1, len(client_accuracy)))
for (client_id, data), color in zip(client_accuracy.items(), colors):
    plt.plot(data["round"], data["accuracy"], 
            # label=f"Client {client_id}", 
            alpha=0.6,
            color=color,
            zorder=2)

# Plot the mean accuracy with enhanced style
line, = plt.plot(mean_accuracy["round"], mean_accuracy["accuracy"], 
         color="red", 
         linewidth=3, 
         linestyle="--",
         label="Global Model Accuracy",
         zorder=3)

# Add marker for maximum mean accuracy
plt.plot(max_accuracy_round, max_mean_accuracy, 'r*', 
         markersize=20,  # 增大星星大小
         zorder=4)

# Add text annotation for max accuracy with enhanced style
plt.annotate(f'Max: {max_mean_accuracy:.4f}\nRound: {int(max_accuracy_round)}',
            xy=(max_accuracy_round, max_mean_accuracy),
            xytext=(20, 20), 
            textcoords='offset points',
            fontsize=16,  # 增大注释字体
            bbox=dict(boxstyle='round,pad=0.5', 
                     fc='yellow', 
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=2),
            arrowprops=dict(arrowstyle='fancy',
                          fc='0.6',
                          ec='none',
                          connectionstyle='arc3,rad=-0.3'))

# Enhance labels and title with larger font sizes
plt.xlabel("Round", fontsize=24, weight='bold')  # 从18增大到24
plt.ylabel("Accuracy", fontsize=24, weight='bold')  # 从18增大到24
plt.title("Accuracy Over Time", fontsize=26, weight='bold', pad=20)  # 从22增大到26

# Enhance legend with larger font
plt.legend(handles=[line],  # 明确指定使用这个线条对象
          labels=["Global Model Accuracy"],
          loc='lower right',
          frameon=True,
          fancybox=True,
          shadow=True,
          fontsize=15)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# Make tick labels bold and larger
plt.xticks(weight='bold', fontsize=18)  # 从14增大到18
plt.yticks(weight='bold', fontsize=18)  # 从14增大到18

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Save with white background (PNG format)
plt.savefig("client_accuracy.png", 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

# 保存为PDF格式
plt.savefig("client_accuracy.pdf", 
            dpi=1200, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            format='pdf')

# Print the statistics
print(f"\nMean Accuracy Maximum: {max_mean_accuracy:.4f} (Round {int(max_accuracy_round)})")
print(f"图表已保存为PNG和PDF格式")