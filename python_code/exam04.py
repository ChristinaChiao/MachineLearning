import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

sns.set_theme(
    style='whitegrid',
    font_scale=1.0,
    rc={
        'font.sans-serif': ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
        'axes.unicode_minus': False,
    },
)
# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 載入資料
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# 2. 基本檢查
print(df.head())
print(df.describe())

# 3. 建立收入分組欄位
q1 = df['MedInc'].quantile(0.33)
q2 = df['MedInc'].quantile(0.66)
df['IncomeLevel'] = pd.cut(
    df['MedInc'],
    bins=[df['MedInc'].min(), q1, q2, df['MedInc'].max()],
    labels=['低收入', '中收入', '高收入'],
    include_lowest=True
)

# 4. 建立畫布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 圖 1：房價直方圖
sns.histplot(df['MedHouseVal'], bins=40, kde=True, color='steelblue', ax=axes[0, 0])
axes[0, 0].set_title('房價中位數分布')
axes[0, 0].set_xlabel('MedHouseVal')

# 圖 2：收入 vs 房價
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal', alpha=0.2, s=15, ax=axes[0, 1])
axes[0, 1].set_title('收入中位數與房價關係')
axes[0, 1].set_xlabel('MedInc')
axes[0, 1].set_ylabel('MedHouseVal')

# 圖 3：收入群組箱型圖
sns.boxplot(
    data=df,
    x='IncomeLevel',
    y='MedHouseVal',
    hue='IncomeLevel',
    palette='Set2',
    legend=False,
    ax=axes[1, 0],
)
axes[1, 0].set_title('不同收入群組的房價比較')
axes[1, 0].set_xlabel('IncomeLevel')
axes[1, 0].set_ylabel('MedHouseVal')

# 圖 4：相關係數熱力圖
corr = df.drop(columns=['IncomeLevel']).corr(numeric_only=True)
sns.heatmap(corr, cmap='Blues', annot=False, square=True, ax=axes[1, 1])
axes[1, 1].set_title('相關係數熱力圖')

plt.tight_layout()

plt.gcf().text(
    0.98,
    0.015,
    "作者：10_蔡宜喬",
    ha="right",
    va="bottom",
    fontsize=10,
    color="black",
)

plt.show()