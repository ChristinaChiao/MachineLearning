import matplotlib.pyplot as plt
import seaborn as sns


# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def main() -> None:
	iris_df = sns.load_dataset('iris')

	fig, ax = plt.subplots(figsize=(7, 5))

	sns.scatterplot(
		data=iris_df,
		x='sepal_length',
		y='sepal_width',
		hue='species',
		palette='Set1',
		ax=ax,
	)
	ax.set_title('花萼長度與寬度散佈圖')
	ax.set_xlabel('花萼長度')
	ax.set_ylabel('花萼寬度')

	plt.tight_layout(rect=[0, 0.08, 1, 1])
	# 在圖框外的右下角顯示作者名字
	fig.text(
		0.98,
		0.02,
		'作者：10_蔡宜喬',
		ha='right',
		va='bottom',
		fontsize=10,
		color='black',
	)
	plt.show()


if __name__ == '__main__':
	main()
