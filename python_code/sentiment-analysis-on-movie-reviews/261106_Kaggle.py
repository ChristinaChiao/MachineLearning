#P11


# ── Stage 1 · 讀取資料 ─────────────────────────────────────
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

data_dir = Path(__file__).resolve().parent
train_df = pd.read_csv(data_dir / "train.tsv.zip", sep="\t")
test_df  = pd.read_csv(data_dir / "test.tsv.zip",  sep="\t")
# train: PhraseId · SentenceId · Phrase · Sentiment
# test : PhraseId · SentenceId · Phrase   (no Sentiment)

# ── Stage 2-1 · 建詞表（Tokenizer 只在 train 上 fit） ─────
train_phrases = train_df["Phrase"].fillna("").astype(str)
test_phrases = test_df["Phrase"].fillna("").astype(str)
y_train = train_df["Sentiment"].to_numpy()   # (156060,) ∈ {0,1,2,3,4}

# ── Stage 2-3 · 切驗證集 ─────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(train_phrases, y_train,
        test_size=0.1, random_state=42, stratify=y_train)
# X_tr=140454  X_val=15606




#P12


# =============================================================
# Stage 3 ── 詞彙與字元 TF-IDF 特徵
# =============================================================
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, sublinear_tf=True,
)
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
)

X_tr_word = word_vectorizer.fit_transform(X_tr)
X_val_word = word_vectorizer.transform(X_val)
X_tr_char = char_vectorizer.fit_transform(X_tr)
X_val_char = char_vectorizer.transform(X_val)
X_tr_features = hstack([X_tr_word, X_tr_char], format="csr")
X_val_features = hstack([X_val_word, X_val_char], format="csr")




#P13


# ── Stage 4 · 驗證模型並以全部資料重訓 ─────────────────────
validation_model = LinearSVC(C=0.5)
validation_model.fit(X_tr_features, y_tr)
val_predictions = validation_model.predict(X_val_features)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Kaggle 評分採 accuracy，使用不加權的分類器。
X_train_word = word_vectorizer.fit_transform(train_phrases)
X_test_word = word_vectorizer.transform(test_phrases)
X_train_char = char_vectorizer.fit_transform(train_phrases)
X_test_char = char_vectorizer.transform(test_phrases)
X_train_features = hstack([X_train_word, X_train_char], format="csr")
X_test_features = hstack([X_test_word, X_test_char], format="csr")

model = LinearSVC(C=0.5)
model.fit(X_train_features, y_train)



# P14


# ── Stage 5 · 對 test.tsv 預測 → 寫出 submission.csv ────────

# 5-1 · 預測五種情緒類別
y_test_pred = model.predict(X_test_features).astype(int)
# shape: (66292,)  每個值 ∈ {0,1,2,3,4}

# 5-3 · 組合 PhraseId 與預測標籤，符合 Kaggle 規範
submission = pd.DataFrame({
    "PhraseId":  test_df["PhraseId"].values,
    "Sentiment": y_test_pred,
})

# 5-4 · 健全性檢查
assert submission.shape == (66292, 2)
assert submission["Sentiment"].between(0, 4).all()
assert submission["PhraseId"].is_unique

# 5-5 · 寫成 CSV（index=False 不要多一個無名欄）
submission.to_csv("submission.csv", index=False)
print(submission.head())

# ── （選用）用 Kaggle CLI 直接上傳 ──────────────────────────
# $ pip install kaggle
# $ kaggle competitions submit \
#       -c sentiment-analysis-on-movie-reviews \
#       -f submission.csv \
#       -m "LSTM baseline, embed=128, units=64"






