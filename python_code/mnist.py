import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. 載入 MNIST 資料集 (這需要一點下載時間)
print("正在下載 MNIST 資料集...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# 2. 資料預處理：將像素值縮放到 0 ~ 1 之間（原本為 0 ~ 255）
# 類神經網路對特徵縮放非常敏感，這能顯著加快收斂速度
X = X / 255.0

# 3. 切分訓練集與測試集 (60,000 張訓練，10,000 張測試)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

# 4. 定義 MLP 模型
# 這裡設定一個隱藏層，包含 100 個神經元
# max_iter 設定最大迭代次數，verbose=True 可以看到訓練過程的 loss 變化
mlp = MLPClassifier(
    hidden_layer_sizes=(100,), 
    activation='relu', 
    solver='adam', 
    max_iter=20, 
    random_state=42,
    verbose=True
)

# 5. 訓練模型
print("開始訓練 MLP 模型...")
mlp.fit(X_train, y_train)

# 6. 預測與評估
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*50)
print(f"測試集準確率 (Accuracy): {accuracy:.4f}")
print("="*50)

# 顯示詳細的各類別分類報告
print("\n詳細分類報告:")
print(classification_report(y_test, y_pred))
