"""Digit Recognizer V2：Windows 本機訓練與 Kaggle 預測。

預設從本程式旁的 data/train.csv、data/test.csv 讀取，結果寫入 results。
範例：python digit_recognizer_v2.py --data-dir "D:/Kaggle/digit-recognizer" --plot
所有相對路徑均以「本程式所在資料夾」為基準，不受終端機目前位置影響。
"""

from pathlib import Path
import argparse
import csv
import multiprocessing
import random
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Windows 預設不建立背景載入程序，降低初次執行與記憶體使用的問題。
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="另存 learning_curves.png")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.step_size) < 1:
        parser.error("epochs、batch-size、step-size 必須大於 0")
    if args.workers < 0 or args.lr <= 0 or not 0 < args.gamma <= 1:
        parser.error("workers 必須 >= 0、lr > 0，且 0 < gamma <= 1")
    if not 0 < args.val_ratio < 1:
        parser.error("val-ratio 必須介於 0 與 1 之間")
    return args


# 延後匯入套件，讓尚未安裝 PyTorch 的環境仍能執行 --help。
# Dataset 類別需位於模組最外層，Windows 多程序才能正確載入。
class DigitDataset:
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image if self.labels is None else (image, self.labels[index])


def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main():
    args = parse_args()
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit(
            f"缺少或無法載入必要套件：{exc}\n"
            "請在執行本程式的同一個 Python 環境安裝套件，詳見 README。"
        ) from exc

    base = Path(__file__).resolve().parent
    data_dir = (base / args.data_dir).resolve()
    output_dir = (base / args.output_dir).resolve()
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到資料檔案：{path}")

    # 固定主要亂數來源；不同硬體或套件版本之間仍可能有微小差異。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch：{torch.__version__}；裝置：{device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU：{props.name}；顯示記憶體：{props.total_memory / 1024**3:.1f} GB")
        print("已偵測 RTX 4060。" if "4060" in props.name else "使用目前可用的 CUDA GPU。")
    else:
        print("CUDA 無法使用，改用 CPU。若有 RTX 4060，請確認 NVIDIA 驅動與 CUDA 版 PyTorch。")

    # 嚴格檢查 Kaggle 欄位與數值，避免讀錯 CSV 後仍繼續訓練。
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    pixels = [f"pixel{i}" for i in range(784)]
    if set(train.columns) != set(["label"] + pixels) or set(test.columns) != set(pixels):
        raise ValueError("train.csv 必須有 label 與 pixel0～pixel783；test.csv 只能有這 784 個像素欄位。")
    if train.empty or test.empty:
        raise ValueError("train.csv 與 test.csv 不可為空。")

    def read_pixels(frame):
        values = frame[pixels].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 255).any():
            raise ValueError("像素必須是 0～255 的有限數值，且不可有缺值。")
        # CNN 輸入維度為 N × C × H × W，灰階圖片只有 1 個通道。
        return torch.from_numpy(values.reshape(-1, 1, 28, 28) / 255.0)

    images, test_images = read_pixels(train), read_pixels(test)
    raw_labels = train["label"].to_numpy(dtype=np.float64)
    if (not np.isfinite(raw_labels).all() or (raw_labels < 0).any()
            or (raw_labels > 9).any() or (raw_labels != np.floor(raw_labels)).any()):
        raise ValueError("label 必須是 0～9 的整數，且不可有缺值。")
    labels = torch.from_numpy(raw_labels.astype(np.int64))
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)), test_size=args.val_ratio,
            random_state=args.seed, stratify=raw_labels,
        )
    except ValueError as exc:
        raise ValueError(f"無法分層切分資料，請確認各類別樣本數及 val-ratio：{exc}") from exc

    # 只對訓練集即時做小幅旋轉、平移與縮放；驗證及測試不做隨機增強。
    augmentation = transforms.RandomAffine(
        degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0,
    )
    train_ds = DigitDataset(images[train_idx], labels[train_idx], augmentation)
    val_ds = DigitDataset(images[val_idx], labels[val_idx])
    test_ds = DigitDataset(test_images)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_options)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.15),  # 28×28 → 14×14
                nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.25),  # 14×14 → 7×7
                nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 10),  # 輸出 logits；CrossEntropyLoss 內部會處理 softmax。
            )

        def forward(self, x):
            return self.network(x)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    def run_epoch(loader, training):
        # eval() 會固定 BatchNorm 統計並關閉 Dropout，驗證時也關閉梯度計算。
        model.train(training)
        loss_sum, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Loss 出現 NaN 或無限值，請檢查資料或降低學習率。")
                if training:
                    loss.backward()
                    optimizer.step()
                n = batch_labels.size(0)
                # 按樣本數加權，最後一個較小 batch 不會造成平均偏差。
                loss_sum += loss.item() * n
                correct += (logits.argmax(1) == batch_labels).sum().item()
                count += n
        return loss_sum / count, correct / count

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.csv"
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    history = []
    fields = ["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"]
    print(f"訓練筆數：{len(train_ds)}；驗證筆數：{len(val_ds)}；測試筆數：{len(test_ds)}")
    print(f"輸出資料夾：{output_dir}（同名結果會覆寫）")
    with history_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(train_loader, True)
            val_loss, val_acc = run_epoch(val_loader, False)
            # 優先選驗證正確率最高的模型；同分時選驗證 loss 較低者。
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
                torch.save(model.state_dict(), best_path)
            row = dict(epoch=epoch, lr=current_lr, train_loss=train_loss,
                       train_accuracy=train_acc, val_loss=val_loss, val_accuracy=val_acc,
                       seconds=time.perf_counter() - started)
            history.append(row)
            writer.writerow(row)
            file.flush()  # 每輪落盤，即使後續中斷仍保留已完成的紀錄。
            print(f"Epoch {epoch:02d}/{args.epochs} | lr={current_lr:.6f} | "
                  f"train loss={train_loss:.4f}, acc={train_acc:.2%} | "
                  f"val loss={val_loss:.4f}, acc={val_acc:.2%} | {row['seconds']:.1f}s")
            scheduler.step()  # 在本輪訓練完成後更新，history 記錄的是該輪實際使用的 lr。

    # 重新載入最佳權重後才預測，避免誤用最後一輪模型。
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch_images in test_loader:
            logits = model(batch_images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(1).cpu().tolist())
    submission_path = output_dir / "submission.csv"
    pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions}).to_csv(
        submission_path, index=False, encoding="utf-8",
    )
    print(f"最佳 Epoch：{best_epoch}；驗證 Accuracy：{best_acc:.2%}")
    print(f"已儲存：{best_path}\n已儲存：{history_path}\n已儲存：{submission_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # 直接輸出圖片，不依賴桌面視窗。
            import matplotlib.pyplot as plt
        except ImportError:
            print("未安裝 matplotlib，略過曲線；模型、歷史及 submission 已完成。")
        else:
            epochs = [row["epoch"] for row in history]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
                ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
                ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
                ax.set(xlabel="Epoch", ylabel=title, title=title)
                ax.grid(alpha=0.3)
                ax.legend()
            fig.tight_layout()
            curve_path = output_dir / "learning_curves.png"
            fig.savefig(curve_path, dpi=160)
            plt.close(fig)
            print(f"已儲存：{curve_path}")


if __name__ == "__main__":
    # Windows 使用 spawn；務必保留 main 保護，避免子程序重複啟動訓練。
    multiprocessing.freeze_support()
    main()
