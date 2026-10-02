from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def main() -> None:
	csv_path = Path(__file__).parent / "2330_Historical_Stock.csv"

	data = pd.read_csv(csv_path, parse_dates=["Date"])
	data = data.sort_values("Date")

	max_close_idx = data["Close"].idxmax()
	max_row = data.loc[max_close_idx]

	plt.figure(figsize=(10, 5))
	plt.plot(data["Date"], data["Close"], color="steelblue", linewidth=2, label="收盤價")

	plt.scatter(
		max_row["Date"],
		max_row["Close"],
		color="red",
		s=80,
		zorder=3,
		label="最高收盤價",
	)
	plt.annotate(
		f"最高價: {max_row['Close']}",
		xy=(max_row["Date"], max_row["Close"]),
		xytext=(15, 15),
		textcoords="offset points",
		arrowprops={"arrowstyle": "->", "color": "red"},
		color="red",
	)

	plt.title("台積電(2330)收盤價折線圖")
	plt.xlabel("日期")
	plt.ylabel("收盤價")
	plt.grid(alpha=0.3)
	plt.legend()
	ax = plt.gca()
	locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
	ax.xaxis.set_major_locator(locator)
	ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
	plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
	plt.tight_layout(rect=[0, 0.08, 1, 1])
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


if __name__ == "__main__":
	main()
