import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

def read_time_stats():
    """读取所有时间统计文件并合并数据"""
    # 读取服务器统计
    server_stats = pd.read_csv("encrypted/server_time_stats.csv")
    server_total = server_stats.groupby('operation')['time'].sum()
    
    # 读取所有客户端统计
    client_files = glob.glob("encrypted/client_*_time_stats.csv")
    client_stats = []
    
    for file in client_files:
        df = pd.read_csv(file)
        client_stats.append(df)
    
    # 合并所有客户端数据
    if client_stats:
        all_client_stats = pd.concat(client_stats)
        client_total = all_client_stats.groupby('operation')['time'].sum()
    else:
        client_total = pd.Series()
    
    # 合并服务器和客户端数据
    all_stats = pd.concat([server_total, client_total])
    return all_stats

def plot_pie_chart(time_stats, total_time):
    """绘制饼图"""
    plt.figure(figsize=(10, 8))
    time_percentages = (time_stats / total_time) * 100
    
    # 使用更优雅的颜色方案
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    # 计算每个操作的百分比和时间
    labels = [f'{op}\n{time:.1f}s ({pct:.1f}%)' 
              for op, time, pct in zip(time_stats.index, time_stats.values, time_percentages)]
    
    # 绘制饼图
    wedges, texts = plt.pie(
        time_percentages, 
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops=dict(
            width=0.7,  # 设置为环形图
            edgecolor='white',
            linewidth=2
        ),
        labeldistance=1.1,  # 标签距离
        textprops={'fontsize': 10, 'weight': 'bold'}
    )
    
    # 添加中心文本（总时间）
    plt.text(0, 0, f'Total Time\n{total_time:.1f}s', 
            ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # 添加标题
    plt.title('Operation Time Distribution', 
             pad=20, size=16, weight='bold')
    
    # 保存图表
    plt.savefig('time_distribution_pie.png', 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    print("Pie chart saved as time_distribution_pie.png")

def plot_bar_chart(time_stats):
    """绘制柱状图"""
    plt.figure(figsize=(12, 7))
    
    # 设置颜色方案
    colors = plt.cm.Set3(np.linspace(0, 1, len(time_stats)))
    
    # 绘制柱状图
    bars = plt.bar(
        time_stats.index,
        time_stats.values,
        color=colors,
        width=0.7,
        edgecolor='black',
        linewidth=1.5,
        zorder=3  # 确保柱子在网格线上面
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.2f}s\n({height/time_stats.sum()*100:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10,
            weight='bold'
        )
    
    # 设置标题和标签
    plt.title('Time Consumption per Operation', pad=20, size=16, weight='bold')
    plt.ylabel('Time (seconds)', size=12, weight='bold')
    
    # 美化x轴
    plt.xticks(rotation=30, ha='right', size=10, weight='bold')
    plt.yticks(weight='bold')
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # 添加轻微的背景色
    plt.gca().set_facecolor('#f8f9fa')
    
    # 添加边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('time_consumption_bar.png', 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    print("Bar chart saved as time_consumption_bar.png")

def print_statistics(time_stats, total_time):
    """打印详细统计信息"""
    time_percentages = (time_stats / total_time) * 100
    
    print("\nDetailed Time Statistics:")
    print("=" * 50)
    print(f"{'Operation':<20} {'Time (s)':>10} {'Percentage':>12}")
    print("-" * 50)
    
    for op in time_stats.index:
        print(f"{op:<20} {time_stats[op]:>10.2f}s {time_percentages[op]:>11.1f}%")
    
    print("=" * 50)
    print(f"Total Time: {total_time:.2f} seconds")

def main():
    # 设置matplotlib的样式
    try:
        import seaborn as sns
        sns.set_style("whitegrid")  # 使用 seaborn 的样式
    except ImportError:
        plt.style.use('classic')  # 如果没有 seaborn，使用 matplotlib 的经典样式
    
    # 设置字体
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # 检查是否存在统计文件
    if not os.path.exists("server_time_stats.csv") and not glob.glob("client_*_time_stats.csv"):
        print("No time statistics files found!")
        return
    
    # 读取数据
    time_stats = read_time_stats()
    total_time = time_stats.sum()
    
    # 绘制图表
    plot_pie_chart(time_stats, total_time)
    plot_bar_chart(time_stats)
    
    # 打印统计信息
    print_statistics(time_stats, total_time)

if __name__ == "__main__":
    main()
