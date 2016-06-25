import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

df_train = pd.read_csv('../data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)

probs = pd.DataFrame()
errors = pd.DataFrame()

X_train = df_train
y_train = df_target
X_train, X_test, y_train, y_test= train_test_split(X_train, y_train, test_size=0.3, random_state=0)

learning_rate = 0.01
n_estimators = 600
max_depth = 6
subsample = 0.9
colsample_bytree = 0.85
min_child_weight = 1  # default

eval_metrics = ['auc']
eval_sets = [(X_train, y_train), (X_test, y_test)]
xgb = XGBClassifier(seed=0, learning_rate=learning_rate, n_estimators=n_estimators,
                    min_child_weight=min_child_weight, max_depth=max_depth,
                    colsample_bytree=colsample_bytree, subsample=subsample)
xgb = xgb.fit(X_train, y_train, eval_metric=eval_metrics, eval_set=eval_sets, verbose=False)

probs['xgb'] = xgb.predict_proba(X_test)[:, -1]

auc = [xgb.evals_result_['validation_%d' % i]['auc'] for i in range(len(eval_sets))]
auc = np.array(auc, dtype=float).T

auc_best_round = np.argmax(auc, axis=0)
auc_best = [auc[auc_best_round[0], 0], auc[auc_best_round[1], 1]]

print 'Best AUC train=%f (round=%d), test=%f (round=%d)' % (auc_best[0], auc_best_round[0], auc_best[1], auc_best_round[1])
