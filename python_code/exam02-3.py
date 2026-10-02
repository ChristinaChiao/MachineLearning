import matplotlib.pyplot as plt
import seaborn as sns


# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def main() -> None:
	iris_df = sns.load_dataset('iris')

	plt.figure(figsize=(8, 5))
	sns.boxplot(data=iris_df, x='species', y='petal_length', palette='Set2')
	plt.title('不同品種鳶尾花的花瓣長度箱型圖')
	plt.xlabel('品種')
	plt.ylabel('花瓣長度')
	plt.tight_layout(rect=[0, 0.08, 1, 1])
	plt.gcf().text(
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
