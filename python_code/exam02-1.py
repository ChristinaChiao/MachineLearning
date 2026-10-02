import matplotlib.pyplot as plt
import numpy as np

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 準備資料
x = np.linspace(0, 10, 100)
y = x ** 2

# 建立圖表
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='y = x^2', color='red', linestyle='--')

# 加入標題與標籤
plt.title('y = x^2 折線圖')
plt.xlabel('X 軸')
plt.ylabel('Y 軸')
plt.legend()

# 在圖的右下角顯示作者名字
plt.gca().text(
	0.98,
	0.02,
	'作者：10_蔡宜喬',
	transform=plt.gca().transAxes,
	ha='right',
	va='bottom',
	fontsize=10,
	color='black',
)

# 顯示圖表
plt.show()