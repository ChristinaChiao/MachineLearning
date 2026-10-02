"""Train a time-validated LightGBM model for IEEE-CIS fraud detection."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DATA = Path(__file__).resolve().parent

train = pd.read_csv(DATA / 'train_transaction.csv')
train_identity = pd.read_csv(DATA / 'train_identity.csv')
test = pd.read_csv(DATA / 'test_transaction.csv')
test_identity = pd.read_csv(DATA / 'test_identity.csv')
test_identity.columns = [column.replace('-', '_') for column in test_identity.columns]

train = train.merge(train_identity, on='TransactionID', how='left')
test = test.merge(test_identity, on='TransactionID', how='left')

y = train.pop('isFraud')
test_ids = test['TransactionID']
X = train.drop(columns=['TransactionID'])
X_test = test.drop(columns=['TransactionID'])

all_features = pd.concat([X, X_test], ignore_index=True)
all_features['TransactionHour'] = (all_features['TransactionDT'] // 3600) % 24

group_keys = {}
for name, columns in {
    'uid_card_addr': ['card1', 'addr1'],
    'uid_card_addr_email': ['card1', 'addr1', 'P_emaildomain'],
    'uid_card_bundle': ['card1', 'card2', 'card3', 'card5'],
    'uid_card_addr_bundle': ['card1', 'addr1', 'card2', 'card3', 'card5'],
}.items():
    group_keys[name] = all_features[columns].astype('string').fillna('__missing__').agg('_'.join, axis=1)

engineered = {}
for name, key in group_keys.items():
    engineered[name + '_count'] = key.map(key.value_counts(dropna=False)).astype('float32')

card_addr = group_keys['uid_card_addr']
amount = all_features['TransactionAmt'].astype('float32')
amount_median = amount.groupby(card_addr).median()
engineered['amount_median_by_card_addr'] = card_addr.map(amount_median).astype('float32')
engineered['amount_mean_by_card_addr'] = card_addr.map(amount.groupby(card_addr).mean()).astype('float32')
engineered['amount_ratio_card_addr'] = (
    amount / (engineered['amount_median_by_card_addr'] + 0.01)
).astype('float32')

transaction_order = pd.DataFrame({
    'uid': card_addr,
    'dt': all_features['TransactionDT'].astype('int64'),
    'row': np.arange(len(all_features)),
}).sort_values(['uid', 'dt'])
transaction_order['previous_dt'] = transaction_order.groupby('uid', sort=False)['dt'].shift(1)
seconds_since_previous = (
    transaction_order['dt'] - transaction_order['previous_dt']
).reindex(all_features.index)
engineered['seconds_since_card_addr'] = seconds_since_previous.fillna(-1).astype('float32')

all_features = pd.concat([all_features, pd.DataFrame(engineered)], axis=1)
X = all_features.iloc[:len(X)].copy()
X_test = all_features.iloc[len(X):].copy()
del all_features, group_keys, engineered, transaction_order, seconds_since_previous

cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
for column in cat_cols:
    values = pd.concat(
        [
            X[column].astype('string').fillna('__missing__'),
            X_test[column].astype('string').fillna('__missing__'),
        ],
        ignore_index=True,
    )
    codes, _ = pd.factorize(values, sort=True)
    X[column] = codes[:len(X)].astype('int32')
    X_test[column] = codes[len(X):].astype('int32')

X = X.astype('float32')
X_test = X_test[X.columns].astype('float32')

cutoff = X['TransactionDT'].quantile(0.8)
train_mask = X['TransactionDT'] < cutoff
valid_mask = ~train_mask

model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    max_bin=127,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
)
model.fit(
    X.loc[train_mask],
    y.loc[train_mask],
    categorical_feature=cat_cols,
    eval_set=[(X.loc[valid_mask], y.loc[valid_mask])],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

valid_prob = model.predict_proba(X.loc[valid_mask])[:, 1]
best_iteration = model.best_iteration_ or model.n_estimators
print(f'Temporal validation AUC: {roc_auc_score(y.loc[valid_mask], valid_prob):.6f}')
print(f'Best iteration: {best_iteration}')

final_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=best_iteration,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    max_bin=127,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
)
final_model.fit(X, y, categorical_feature=cat_cols)

submission = pd.DataFrame(
    {'TransactionID': test_ids, 'isFraud': final_model.predict_proba(X_test)[:, 1]}
)
submission.to_csv(DATA / 'submission.csv', index=False)
print(f'Saved {len(submission):,} predictions to {DATA / "submission.csv"}')