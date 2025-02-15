import numpy as np
import matplotlib.pyplot as plt
import os

# 设置阈值
thresholds = np.arange(0.50, 1.0, 0.05)

# 创建图形
plt.figure(figsize=(12, 8))

# 读取result_of_threshold文件夹下的所有txt文件
folder_path = "result_of_threshold"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        results = np.loadtxt(file_path)

        # 确保结果长度与阈值长度一致
        if len(results) != len(thresholds):
            print(
                f"Warning: {filename} has {len(results)} values, expected {len(thresholds)}. Skipping this file."
            )
            continue

        # 绘制折线图
        plt.plot(
            thresholds, results, "-o", linewidth=2, markersize=6, label=filename[:-4]
        )
# 设置标题和标签
plt.title("F1 Score of Different Threshold ", fontsize=16)
plt.xlabel("Threshold", fontsize=14)
plt.ylabel("F1 Score", fontsize=14)

# 添加网格
plt.grid(True, linestyle="--", alpha=0.7)

# 设置x轴刻度
plt.xticks(thresholds)

# 添加图例
plt.legend(fontsize=10, loc="best")

# 调整布局
plt.tight_layout()
# 保存图形
plt.savefig("threshold_f1_plot_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
