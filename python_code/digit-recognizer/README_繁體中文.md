# Digit Recognizer V2 Windows 使用說明

將 `digit_recognizer_v2.py` 放到自己的專案資料夾。請先從 Kaggle Digit Recognizer 下載並解壓縮 `train.csv` 與 `test.csv`。程式不會自動下載資料。

## 1. 安裝套件

請使用實際執行程式的 Python 安裝套件，避免安裝到其他環境。以先前的虛擬環境為例：

```powershell
D:\venv1\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib
```

PyTorch 與 torchvision 請到官方安裝頁，選擇 Windows、Pip、Python 與適用的 CUDA 選項，再將產生指令中的 `pip3` 換成 `D:\venv1\Scripts\python.exe -m pip` 執行。務必使用相容的一組 torch / torchvision。官方選擇器：https://pytorch.org/get-started/locally/

檢查目前環境：

```powershell
D:\venv1\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

RTX 4060 需有適用的 NVIDIA 驅動與 CUDA 版 PyTorch。程式會自動使用第一張可用 CUDA GPU，並顯示名稱及記憶體；CUDA 不可用時會改用 CPU。無須在程式硬編碼 RTX 4060 名稱。

## 2. 準備路徑並執行

預設結構：

```text
你的專案資料夾/
  digit_recognizer_v2.py
  data/
    train.csv
    test.csv
```

在程式所在資料夾執行：

```powershell
D:\venv1\Scripts\python.exe .\digit_recognizer_v2.py --plot
```

或指定本機絕對路徑（有空白也可，請保留引號）：

```powershell
D:\venv1\Scripts\python.exe .\digit_recognizer_v2.py --data-dir "D:\Kaggle\digit-recognizer" --output-dir "D:\Kaggle\digit-results-v2" --epochs 20 --batch-size 128 --plot
```

若使用 PowerShell 且 Python 執行檔路徑有空白，在引號前加 `&`。相對資料與輸出路徑一律相對於 Python 程式位置。預設輸出到程式旁的 `results`，同名結果會覆寫；要保存多次實驗請每次指定不同的 `--output-dir`。顯示記憶體不足時可將 `--batch-size` 改為 64 或 32。Windows 的 `--workers` 預設為 0，可先確認正常執行後再調高。

## 3. 輸出內容

| 檔案 | 用途 |
| --- | --- |
| best_model.pth | 驗證 Accuracy 最高的模型權重；同分時取驗證 Loss 較低者 |
| submission.csv | Kaggle 提交檔，只有 ImageId、Label，ImageId 從 1 開始，順序與 test.csv 相同 |
| history.csv | 每輪 epoch、lr、train_loss、train_accuracy、val_loss、val_accuracy、seconds |
| learning_curves.png | 加上 --plot 時產生的 Loss 與 Accuracy 曲線 |

history.csv 中 Accuracy 為 0～1；終端機顯示為百分比。訓練 Loss/Accuracy 來自啟用資料增強與 Dropout 的訓練過程，驗證使用 eval 模式，因此兩者不能直接視為相同條件下的評估。

程式先以固定亂數種子分層切出 10% 驗證資料，只對訓練資料做 RandomAffine。使用 BatchNorm + Dropout CNN、Adam 與 StepLR，預設每 5 輪將學習率乘以 0.5。預測前會重新載入最佳權重。此版保留驗證集，不會再用全量資料重新訓練。

best_model.pth 僅含 state_dict 權重，不含優化器狀態，適用於相同 CNN 結構的推論，並非中斷續訓檢查點。程式使用官方建議的 state_dict 儲存／載入方式：https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

可用 `--help` 查看所有參數。學習曲線是選用功能；未安裝 matplotlib 時會略過，其他輸出仍保留。
