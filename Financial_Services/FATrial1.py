import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import auc
from sklearn import metrics

df = pd.read_csv('/Users/cbrandenburg/documents/ie/courses/term3/github/ie-group-d-term3/financial_services/train.csv')
#df_train = df_train[0:46020]
#df_test = df_train[-30000:]
msk = np.random.rand(len(df)) < 0.7
df_train = df[msk]
df_test = df[~msk]
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)
df_test_target = df_test['TARGET']
df_test = df_test.drop(['TARGET','ID'], axis = 1)




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
print("Fitting the model")
xgb = xgb.fit(X_train, y_train, eval_metric=eval_metrics, eval_set=eval_sets, verbose=False)
    
print("Predicting Probabilities")
probs['xgb'] = xgb.predict_proba(X_test)[:, -1]

print("Computing AUC")
auc_test = [xgb.evals_result_['validation_%d' % i]['auc'] for i in range(len(eval_sets))]
auc_test = np.array(auc_test, dtype=float).T

auc_best_round = np.argmax(auc_test, axis=0)
auc_best = [auc_test[auc_best_round[0], 0], auc_test[auc_best_round[1], 1]]

print('Best AUC train=%f (round=%d), test=%f (round=%d)' % (auc_best[0], auc_best_round[0], auc_best[1], auc_best_round[1]))
print('Validation')
test_probs = pd.DataFrame()
test_probs['xgb_valid'] = xgb.predict_proba(df_test)[:,-1]
print(test_probs['xgb_valid'].head())

fpr, tpr, thresholds = metrics.roc_curve(df_test_target, test_probs, pos_label=1)

a = float(auc(fpr,tpr))
print(a)

#plot ROC
#preds = clf.predict_proba(Xtest)[:,1]
#fpr, tpr, _ = metrics.roc_curve(ytest, preds)

#df_new = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
#ggplot(df_new, aes(x='fpr', y='tpr')) +\
    #geom_line() +\
    #geom_abline(linetype='dashed')

#print(auc(test_probs['xgb_valid'],df_test_target, reorder=True))


