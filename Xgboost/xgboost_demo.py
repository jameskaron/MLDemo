# 保险赔偿预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import cross_val_score

from scipy import stats
import seaborn as sns
from copy import deepcopy

train = pd.read_csv(
    '/Users/momokohong/Documents/James_Document/Python/CSDN机器学习课程课件/Python数据分析与机器学习实战/Xgboost/train.csv')
test = pd.read_csv('/Users/momokohong/Documents/James_Document/Python/CSDN机器学习课程课件/Python数据分析与机器学习实战/Xgboost/test.csv')

# 先来瞅瞅数据长啥样
print(train.shape)

print('First 20 columns:', list(train.columns[:20]))

print('Last 20 columns:', list(train.columns[-20:]))

print(train.describe())

# 查看缺失值
# 绝大多数情况下，我们都需要对缺失值进行处理
pd.isnull(train).values.any()

#  Continuous vs caterogical features
train.info()

cat_features = list(train.select_dtypes(include=['object']).columns)
print("Categorical: {} features".format(len(cat_features)))

cont_features = [cont for cont in list(train.select_dtypes(
    include=['float64', 'int64']).columns) if cont not in ['loss', 'id']]
print("Continuous: {} features".format(len(cont_features)))

id_col = list(train.select_dtypes(include=['int64']).columns)
print("A column of int64: {}".format(id_col))

# 类别值中属性的个数
cat_uniques = []
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))

uniq_values_in_categories = pd.DataFrame.from_items([('cat_name', cat_features), ('unique_values', cat_uniques)])

print(uniq_values_in_categories.head())

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 5)
ax1.hist(uniq_values_in_categories.unique_values, bins=50)
ax1.set_title('Amount of categorical features with X distinct values')
ax1.set_xlabel('Distinct values in a feature')
ax1.set_ylabel('Features')
ax1.annotate('A feature with 326 vals', xy=(322, 2), xytext=(200, 38), arrowprops=dict(facecolor='black'))

ax2.set_xlim(2, 30)
ax2.set_title('Zooming in the [0,30] part of left histogram')
ax2.set_xlabel('Distinct values in a feature')
ax2.set_ylabel('Features')
ax2.grid(True)
ax2.hist(uniq_values_in_categories[uniq_values_in_categories.unique_values <= 30].unique_values, bins=30)
ax2.annotate('Binary features', xy=(3, 71), xytext=(7, 71), arrowprops=dict(facecolor='black'))
plt.show()

# 赔偿值
plt.figure(figsize=(16, 8))
plt.plot(train['id'], train['loss'])
plt.title('Loss values per id')
plt.xlabel('id')
plt.ylabel('loss')
plt.legend()
plt.show()

# 损失值中有几个显著的峰值表示严重事故。这样的数据分布，使得这个功能非常扭曲导致的回归表现不佳。
# 基本上，偏度度量了实值随机变量的均值分布的不对称性。让我们计算损失的偏度：
stats.mstats.skew(train['loss']).data

print(stats.mstats.skew(train['loss']).data)

print(stats.mstats.skew(np.log(train['loss'])).data)

# 连续值特征
train[cont_features].hist(bins=50, figsize=(16, 12))

# 特征之间的相关性
plt.subplots(figsize=(16, 9))
correlation_mat = train[cont_features].corr()
sns.heatmap(correlation_mat, annot=True)
plt.show()

# Part 2, XGBoost
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings('ignore')

# This may raise an exception in earlier versions of Jupyter

# 数据预处理
train = pd.read_csv(
    '/Users/momokohong/Documents/James_Document/Python/CSDN机器学习课程课件/Python数据分析与机器学习实战/Xgboost/train.csv')

# 做对数转换
train['log_loss'] = np.log(train['loss'])

# 数据分成连续和离散特征
features = [x for x in train.columns if x not in ['id', 'loss', 'log_loss']]

cat_features = [x for x in train.select_dtypes(
    include=['object']).columns if x not in ['id', 'loss', 'log_loss']]
num_features = [x for x in train.select_dtypes(
    exclude=['object']).columns if x not in ['id', 'loss', 'log_loss']]

print("Categorical features:", len(cat_features))
print("Numerical features:", len(num_features))

# And use a label encoder for categorical features:
ntrain = train.shape[0]

train_x = train[features]
train_y = train['log_loss']

for c in range(len(cat_features)):
    train_x[cat_features[c]] = train_x[cat_features[c]].astype('category').cat.codes

print("Xtrain:", train_x.shape)
print("ytrain:", train_y.shape)


# Simple XGBoost Model
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))


# Mode
dtrain = xgb.DMatrix(train_x, train['log_loss'])

# Xgboost参数
# 'booster':'gbtree',
# 'objective': 'multi:softmax', 多分类的问题
# 'num_class':10, 类别数，与 multisoftmax 并用
# 'gamma':损失下降多少才进行分裂
# 'max_depth':12, 构建树的深度，越大越容易过拟合
# 'lambda':2, 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# 'subsample':0.7, 随机采样训练样本
# 'colsample_bytree':0.7, 生成树时进行的列采样
# 'min_child_weight':3, 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
# 'silent':0 ,设置成1则没有运行信息输出，最好是设置为0.
# 'eta': 0.007, 如同学习率
# 'seed':1000,
# 'nthread':7, cpu 线程数
xgb_params = {
    'seed': 0,
    'eta': 0.1,
    'colsample_bytree': 0.5,
    'silent': 1,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'max_depth': 5,
    'min_child_weight': 3
}

# 使用交叉验证 xgb.cv
bst_cv1 = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=3, seed=0,
                 feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)

print('CV score:', bst_cv1.iloc[-1, :]['test-mae-mean'])

plt.figure()
bst_cv1[['train-mae-mean', 'test-mae-mean']].plot()
plt.show()

# 建立100个树模型
bst_cv2 = xgb.cv(xgb_params, dtrain, num_boost_round=100,
                 nfold=3, seed=0, feval=xg_eval_mae, maximize=False,
                 early_stopping_rounds=10)

print('CV score:', bst_cv2.iloc[-1, :]['test-mae-mean'])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 4)

ax1.set_title('100 rounds of training')
ax1.set_xlabel('Rounds')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.plot(bst_cv2[['train-mae-mean', 'test-mae-mean']])
ax1.legend(['Training Loss', 'Test Loss'])

ax2.set_title('60 last rounds of training')
ax2.set_xlabel('Rounds')
ax2.set_ylabel('Loss')
ax2.grid(True)
ax2.plot(bst_cv2.iloc[40:][['train-mae-mean', 'test-mae-mean']])
ax2.legend(['Training Loss', 'Test Loss'])
plt.show()


# XGBoost 参数调节
# Step 1: 选择一组初始参数
# Step 2: 改变 max_depth 和 min_child_weight.
# Step 3: 调节 gamma 降低模型过拟合风险.
# Step 4: 调节 subsample 和 colsample_bytree 改变数据采样策略.
# Step 5: 调节学习率 eta.

class XGBoostRegressor(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'silent': 1, 'objective': 'reg:linear', 'seed': 0})

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                             feval=xg_eval_mae, maximize=False)

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)

    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)
        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=nfold, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
        return cv_rounds.iloc[-1, :]

    def plot_feature_importances(self):
        feat_imp = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

def mae_score(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))

mae_scorer = make_scorer(mae_score, greater_is_better=False)

bst = XGBoostRegressor(eta=0.1, colsample_bytree=0.5, subsample=0.5,
                       max_depth=5, min_child_weight=3, num_boost_round=50)

print(bst.kfold(train_x, train_y, nfold=5))


# Step 1: 基准模型
# Step 2: 树的深度与节点权重
# 这些参数对xgboost性能影响最大，因此，他们应该调整第一。我们简要地概述它们：
# max_depth: 树的最大深度。增加这个值会使模型更加复杂，也容易出现过拟合，深度3-10是合理的。
# min_child_weight: 正则化参数. 如果树分区中的实例权重小于定义的总和，则停止树构建过程。

xgb_param_grid = {'max_depth': list(range(4,9)), 'min_child_weight': list((1,3,6))}
xgb_param_grid['max_depth']

grid = GridSearchCV(XGBoostRegressor(eta=0.1, num_boost_round=50, colsample_bytree=0.5, subsample=0.5),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)

grid.fit(train_x, train_y.values)

grid.grid_scores_, grid.best_params_, grid.best_score_

def convert_grid_scores(scores):
    _params = []
    _params_mae = []
    for i in scores:
        _params.append(i[0].values())
        _params_mae.append(i[1])
    params = np.array(_params)
    grid_res = np.column_stack((_params,_params_mae))
    return [grid_res[:,i] for i in range(grid_res.shape[1])]

_,scores =  convert_grid_scores(grid.grid_scores_)
scores = scores.reshape(5,3)

plt.figure(figsize=(10,5))
cp = plt.contourf(xgb_param_grid['min_child_weight'], xgb_param_grid['max_depth'], scores, cmap='BrBG')
plt.colorbar(cp)
plt.title('Depth / min_child_weight optimization')
plt.annotate('We use this', xy=(5.95, 7.95), xytext=(4, 7.5), arrowprops=dict(facecolor='white'), color='white')
plt.annotate('Good for depth=7', xy=(5.98, 7.05),
             xytext=(4, 6.5), arrowprops=dict(facecolor='white'), color='white')
plt.xlabel('min_child_weight')
plt.ylabel('max_depth')
plt.grid(True)
plt.show()

# Step 3: 调节 gamma去降低过拟合风险
xgb_param_grid = {'gamma':[ 0.1 * i for i in range(0,5)]}

grid = GridSearchCV(XGBoostRegressor(eta=0.1, num_boost_round=50, max_depth=8, min_child_weight=6,
                                        colsample_bytree=0.5, subsample=0.5),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)

grid.fit(train_x, train_y.values)

print(grid.grid_scores_, grid.best_params_, grid.best_score_)


# Step 4: 调节样本采样方式 subsample 和 colsample_bytree
xgb_param_grid = {'subsample':[ 0.1 * i for i in range(6,9)],
                      'colsample_bytree':[ 0.1 * i for i in range(6,9)]}


grid = GridSearchCV(XGBoostRegressor(eta=0.1, gamma=0.2, num_boost_round=50, max_depth=8, min_child_weight=6),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
grid.fit(train_x, train_y.values)

print(grid.grid_scores_, grid.best_params_, grid.best_score_)

_, scores =  convert_grid_scores(grid.grid_scores_)
scores = scores.reshape(3,3)

plt.figure(figsize=(10,5))
cp = plt.contourf(xgb_param_grid['subsample'], xgb_param_grid['colsample_bytree'], scores, cmap='BrBG')
plt.colorbar(cp)
plt.title('Subsampling params tuning')
plt.annotate('Optimum', xy=(0.895, 0.6), xytext=(0.8, 0.695), arrowprops=dict(facecolor='black'))
plt.xlabel('subsample')
plt.ylabel('colsample_bytree')
plt.grid(True)
plt.show()

# Step 5: 减小学习率并增大树个数
# 参数优化的最后一步是降低学习速度，同时增加更多的估计量
# First, we plot different learning rates for a simpler model (50 trees):
xgb_param_grid = {'eta': [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=50, gamma=0.2, max_depth=8, min_child_weight=6,
                                     colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)

grid.fit(train_x, train_y.values)

print(grid.grid_scores_, grid.best_params_, grid.best_score_)

eta, y = convert_grid_scores(grid.grid_scores_)
plt.figure(figsize=(10,4))
plt.title('MAE and ETA, 50 trees')
plt.xlabel('eta')
plt.ylabel('score')
plt.plot(eta, -y)
plt.grid(True)
plt.show()

# 现在我们把树的个数增加到100

xgb_param_grid = {'eta':[0.5,0.4,0.3,0.2,0.1,0.075,0.05,0.04,0.03]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=100, gamma=0.2, max_depth=8, min_child_weight=6,
                                        colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)

grid.fit(train_x, train_y.values)

print(grid.grid_scores_, grid.best_params_, grid.best_score_)

eta, y = convert_grid_scores(grid.grid_scores_)
plt.figure(figsize=(10,4))
plt.title('MAE and ETA, 100 trees')
plt.xlabel('eta')
plt.ylabel('score')
plt.plot(eta, -y)
plt.grid(True)
plt.show()

# 学习率低一些的效果更好
# 200
xgb_param_grid = {'eta':[0.09,0.08,0.07,0.06,0.05,0.04]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=200, gamma=0.2, max_depth=8, min_child_weight=6,
                                        colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)

grid.fit(train_x, train_y.values)

print(grid.grid_scores_, grid.best_params_, grid.best_score_)

eta, y = convert_grid_scores(grid.grid_scores_)
plt.figure(figsize=(10,4))
plt.title('MAE and ETA, 200 trees')
plt.xlabel('eta')
plt.ylabel('score')
plt.plot(eta, -y)
plt.grid(True)
plt.show()

# Final XGBoost model
bst = XGBoostRegressor(num_boost_round=200, eta=0.07, gamma=0.2, max_depth=8, min_child_weight=6,
                                        colsample_bytree=0.6, subsample=0.9)
cv = bst.kfold(train_x, train_y, nfold=5)

print(cv)

# 我们看到200棵树最好的ETA是0.07。正如我们所预料的那样，ETA和num_boost_round依赖关系不是线性的，但是有些关联。
# 们花了相当长的一段时间优化xgboost. 从初始值: 1219.57. 经过调参之后达到 MAE=1171.77.
# 我们还发现参数之间的关系ETA和num_boost_round：
# 100 trees, eta=0.1: MAE=1152.247
# 200 trees, eta=0.07: MAE=1145.92
# `XGBoostRegressor(num_boost_round=200, gamma=0.2, max_depth=8, min_child_weight=6, colsample_bytree=0.6, subsample=0.9, eta=0.07).