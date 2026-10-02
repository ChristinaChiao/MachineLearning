import matplotlib.pyplot as plt

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 準備資料
fruit_sales = {
	'蘋果': 120,
	'香蕉': 95,
	'芒果': 80,
	'藍莓': 60,
	'草莓': 110,
}
fruits = list(fruit_sales.keys())
sales = list(fruit_sales.values())

# 建立圖表
plt.figure(figsize=(8, 4))
plt.bar(fruits, sales, color='skyblue')
plt.subplots_adjust(bottom=0.2)

# 加入標題與標籤
plt.title('水果銷售量長條圖')
plt.xlabel('水果名稱')
plt.ylabel('銷售量')

# 在圖框外的右下角顯示作者名字
plt.gcf().text(
	0.98,
	0.04,
	'作者：10_蔡宜喬',
	ha='right',
	va='bottom',
	fontsize=10,
	color='black',
)


# 顯示圖表
plt.show()