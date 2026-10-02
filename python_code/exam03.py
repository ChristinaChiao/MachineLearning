import csv
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def load_iris_data(file_path: Path) -> List[Dict[str, Union[float, str]]]:
	rows: List[Dict[str, Union[float, str]]] = []

	with file_path.open("r", encoding="utf-8") as file_handle:
		reader = csv.reader(file_handle)
		for row in reader:
			if len(row) != 5:
				continue
			try:
				rows.append(
					{
						"sepal_length": float(row[0]),
						"sepal_width": float(row[1]),
						"petal_length": float(row[2]),
						"petal_width": float(row[3]),
						"species": row[4].strip(),
					}
				)
			except ValueError:
				continue

	return rows


def plot_species_scatter(rows: List[Dict[str, Union[float, str]]]) -> None:
	species_style = {
		"Iris-setosa": {"label": "山鳶尾", "color": "#1f77b4"},
		"Iris-versicolor": {"label": "變色鳶尾", "color": "#ff7f0e"},
		"Iris-virginica": {"label": "維吉尼亞鳶尾", "color": "#2ca02c"},
	}

	plt.figure(figsize=(8, 5))
	for species_key, style in species_style.items():
		species_rows = [item for item in rows if item["species"] == species_key]
		x_values = [item["petal_length"] for item in species_rows]
		y_values = [item["petal_width"] for item in species_rows]
		plt.scatter(x_values, y_values, label=style["label"], color=style["color"], alpha=0.8)

    
	plt.title("鳶尾花品花瓣長度VS花瓣寬度")
	plt.xlabel("花瓣長度")
	plt.ylabel("花瓣寬度")
	plt.legend()
	plt.grid(alpha=0.2)
	plt.tight_layout(rect=[0, 0.06, 1, 1])
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


def main() -> None:
	data_path = Path(__file__).parent / "46_iris" / "iris.data"
	iris_rows = load_iris_data(data_path)
	plot_species_scatter(iris_rows)

if __name__ == "__main__":
	main()
