import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('data/daily_data/SPY_intraday_2024-04-16.csv')
data = data.drop(columns=data.columns[0])
print(data.head())
# 计算相关系数矩阵
corr_matrix = data.corr()

# 绘制热力图
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.savefig('data/analysis_pic/SPY_intraday_2024-04-16.png')
# plt.show()

# 设置相关系数阈值
threshold = 0.8

# 获取高度相关的特征对
highly_correlated_pairs = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > threshold]

# 打印高度相关的特征对
for pair in highly_correlated_pairs:
    print(f"Highly correlated pair: {pair} with correlation coefficient {corr_matrix.loc[pair[0], pair[1]]:.2f}")

# 假设决定删除部分冗余特征
# features_to_drop = ['BB_Width', 'MACD_Signal', 'MACD_Hist']  # 示例特征，根据分析结果调整

# 删除冗余特征
# data = data.drop(columns=features_to_drop)

# 保存处理后的数据
# data.to_csv('/mnt/data/SPY_intraday_2024-04-16_processed.csv', index=False)
