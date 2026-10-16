# ====== 1. 匯入套件 ======
import time
import random
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from lion_pytorch import Lion
import torch_optimizer as torch_optim

try:
    SophiaG = importlib.import_module("Sophia").SophiaG
except ModuleNotFoundError as error:
    if error.name != "Sophia":
        raise
    try:
        SophiaG = importlib.import_module("sophia_optimizer").SophiaG
    except ModuleNotFoundError as error:
        if error.name != "sophia_optimizer":
            raise
        try:
            SophiaG = importlib.import_module("sophia_opt").SophiaG
        except ModuleNotFoundError as error:
            if error.name != "sophia_opt":
                raise
            SophiaG = None

# 固定亂數種子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用裝置：", device)


# ====== 2. 載入 MNIST ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)

# 使用獨立的 Generator，讓各優化器的資料順序一致
def make_train_loader():
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

test_loader = DataLoader(
    test_set,
    batch_size=1000,
    shuffle=False,
    num_workers=0
)


# ====== 3. CNN 模型 ======
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ====== 4. 定義四種優化器 ======
def make_optimizer(name, params):
    if name == "AdamW":
        return torch.optim.AdamW(
            params, lr=0.001, weight_decay=0.01
        )

    elif name == "Lion":
        # Lion 通常使用比 AdamW 更小的學習率
        return Lion(
            params, lr=0.0001, weight_decay=0.01
        )

    elif name == "Sophia":
        return SophiaG(
            params,
            lr=0.0001,
            betas=(0.965, 0.99),
            rho=0.04,
            weight_decay=0.01
        )

    elif name == "Shampoo":
        return torch_optim.Shampoo(
            params,
            lr=0.001,
            momentum=0.0,
            weight_decay=0.0,
            epsilon=1e-4,
            update_freq=1
        )

    else:
        raise ValueError(f"未知的優化器：{name}")


# ====== 5. 訓練一個 Epoch ======
def train_one_epoch(model, loader, optimizer, name):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()

        # Sophia-G 需要額外更新 Hessian 對角估計值
        if name == "Sophia":
            if not hasattr(optimizer, "batch_count"):
                optimizer.batch_count = 0

            # 每 10 個 mini-batch 更新一次估計值
            if optimizer.batch_count % 10 == 0:
                optimizer.update_hessian()

            # bs 使用實際 batch size
            optimizer.step(bs=x.size(0))
            optimizer.batch_count += 1
        else:
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ====== 6. 評估模型 ======
def evaluate(model, loader):
    model.eval()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, accuracy


# ====== 7. 四種優化器進行比較 ======
optim_names = ["AdamW", "Lion", "Sophia", "Shampoo"]
EPOCHS = 10

results = {}

for name in optim_names:
    print(f"\n===== Training with {name} =====")

    if name == "Sophia" and SophiaG is None:
        print("略過 Sophia：請安裝 Sophia optimizer 套件後再執行。")
        continue

    # 每個優化器使用相同的模型初始化
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = CNN().to(device)
    optimizer = make_optimizer(name, model.parameters())
    train_loader = make_train_loader()

    train_losses = []
    test_losses = []
    test_accs = []

    start_time = time.time()

    for ep in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, name
        )

        test_loss, test_acc = evaluate(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(
            f"Epoch {ep:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

    elapsed = time.time() - start_time

    results[name] = {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "acc": test_accs,
        "time": elapsed
    }

    print(f"{name} 訓練時間：{elapsed:.2f} 秒")


# ====== 8. 繪製比較圖 ======
fig, axes = plt.subplots(1, 3, figsize=(19, 5))

for name, hist in results.items():
    epochs = range(1, EPOCHS + 1)

    axes[0].plot(epochs, hist["train_loss"], marker=".",
                 label=name)
    axes[1].plot(epochs, hist["acc"], marker=".",
                 label=name)
    axes[2].plot(epochs, hist["test_loss"], marker=".",
                 label=name)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[1].set_title("Test Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)

axes[2].set_title("Test Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")

for ax in axes:
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimizer_comparison.png", dpi=150)
plt.show()


# ====== 9. 輸出最終比較表 ======
print("\n===== 最終比較結果 =====")
print(
    f"{'Optimizer':<12}"
    f"{'Final Train Loss':>18}"
    f"{'Final Test Loss':>17}"
    f"{'Test Acc (%)':>15}"
    f"{'Time (s)':>12}"
)

for name, hist in results.items():
    print(
        f"{name:<12}"
        f"{hist['train_loss'][-1]:>18.4f}"
        f"{hist['test_loss'][-1]:>17.4f}"
        f"{hist['acc'][-1] * 100:>15.2f}"
        f"{hist['time']:>12.2f}"
    )

# 依最後一個 Epoch 的測試準確率排序
ranking = sorted(
    results.items(),
    key=lambda item: item[1]["acc"][-1],
    reverse=True
)

print("\n依最終測試準確率排序：")
for i, (name, hist) in enumerate(ranking, start=1):
    print(
        f"{i}. {name}: "
        f"{hist['acc'][-1] * 100:.2f}%"
    )