import os.path
import pandas as pd
import numpy as np
import xgboost as xgb

base_path = "/Users/momokohong/Documents/James_Document/Python/CSDN机器学习课程课件/Python数据分析与机器学习实战/第二十四章：商品销售额回归分析/data"

# 数据读取

navigation = pd.read_csv(os.path.join(base_path, 'navigation.csv'))
sales = pd.read_csv(os.path.join(base_path, 'sales.csv'))
train = pd.read_csv(os.path.join(base_path, 'train.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))
vimages = pd.read_csv(os.path.join(base_path, 'vimages.csv'))

# print(train.head())
# print(sales.head())

# 获取不同颜色产品的平均指标
product_descriptor = ['product_type', 'product_gender', 'macro_function',
                      'function', 'sub_function', 'model', 'aesthetic_sub_line', 'macro_material',
                      'month']
product_target_sum = train.groupby(product_descriptor)['target'].sum().reset_index(name = 'sum_target')
product_target_count = train.groupby(product_descriptor)['target'].count().reset_index(name = 'count_target')
product_target_stats = pd.merge(product_target_sum,product_target_count,on=product_descriptor)

train = train.merge(product_target_stats,on=product_descriptor,how='left')
test = test.merge(product_target_stats,on=product_descriptor,how='left')

train['mean_target'] = (train['sum_target'] - train['target'])/(train['count_target']-1)
test['mean_target'] = test['sum_target']/test['count_target']

print(train.head())

# 错误操作:
train.drop(['count_target','sum_target'],axis=1) #这块需要指定inplace 没指定相当于打印不改变DF
test.drop(['count_target','sum_target'],axis=1)
# print(train.head())

# 正确操作
train.drop(['count_target','sum_target'],axis=1,inplace =True)
test.drop(['count_target','sum_target'],axis=1,inplace =True)

print(train.head())

# 统计同款的数量
count_vec_cols = ['macro_function', 'function', 'sub_function', 'model',
                  'aesthetic_sub_line', 'macro_material', 'color']

for col in count_vec_cols:
    tmp = pd.DataFrame(
        {'sku_hash': pd.concat([train['sku_hash'], test['sku_hash']]), col: pd.concat([train[col], test[col]])})
    tmp = pd.DataFrame(tmp.groupby(col)['sku_hash'].count()).reset_index()
    tmp.columns = [col, col + '_count']

    train = train.merge(tmp, on=col, how='left')
    test = test.merge(tmp, on=col, how='left')

print(train.head())

# 统计不同访问源的数量
traffic_source_views = navigation.groupby(['sku_hash','traffic_source'])['page_views'].sum().reset_index()
traffic_source_views[:5]

traffic_source_views = traffic_source_views.pivot(index = 'sku_hash',columns = 'traffic_source',values = 'page_views').reset_index()
traffic_source_views.columns = ['sku_hash',
                                'page_views_nav1', 'page_views_nav2', 'page_views_nav3',
                                'page_views_nav4', 'page_views_nav5', 'page_views_nav6']

print(traffic_source_views.head())

# 统计不同类型的销售数量
type_sales = sales.groupby(['sku_hash','type'])['sales_quantity'].sum().reset_index()
type_sales = type_sales.pivot(index = 'sku_hash',columns = 'type',values = 'sales_quantity').reset_index()
type_sales.columns = ['sku_hash', 'sales_quantity_type1', 'sales_quantity_type2']

print(type_sales.head())

# 统计不同地区情况
zone_sales = sales.groupby(['sku_hash','zone_number'])['sales_quantity'].sum().reset_index()

zone_sales = zone_sales.pivot(index = 'sku_hash',columns = 'zone_number',values = 'sales_quantity').reset_index()
zone_sales.columns = ['sku_hash',
                      'sales_quantity_zone1', 'sales_quantity_zone2', 'sales_quantity_zone3',
                      'sales_quantity_zone4', 'sales_quantity_zone5']

print(zone_sales.head())

# 统计各种网络情况
navigation_stats = navigation.groupby(['sku_hash'])['page_views'].sum().reset_index(name='page_views')
sales_stats = sales.groupby(['sku_hash'])['sales_quantity','TotalBuzzPost', 'TotalBuzz','NetSentiment', 'PositiveSentiment', 'NegativeSentiment', 'Impressions'].sum().reset_index()

print(navigation_stats.head())
print(sales_stats.head())

# 划分数据集
train['idx'] = pd.Categorical(train.sku_hash).codes
train['idx'] = train['idx'] % 5
print(train.head())

# 整合数据集
X = train.copy()
X = X.merge(navigation_stats, on = 'sku_hash', how = 'left')
X = X.merge(sales_stats, on = 'sku_hash', how = 'left')
X = X.merge(traffic_source_views, on = 'sku_hash', how = 'left')
X = X.merge(type_sales, on = 'sku_hash', how = 'left')
X = X.merge(zone_sales, on = 'sku_hash', how = 'left')
print(X.head())

X = train.copy()
X = X.merge(navigation_stats, on = 'sku_hash', how = 'left')
X = X.merge(sales_stats, on = 'sku_hash', how = 'left')
X = X.merge(traffic_source_views, on = 'sku_hash', how = 'left')
X = X.merge(type_sales, on = 'sku_hash', how = 'left')
X = X.merge(zone_sales, on = 'sku_hash', how = 'left')

X.loc[X.product_type=='Accessories','product_type'] = '0'
X.loc[X.product_type=='Leather Goods','product_type'] = '1'
X.product_type = X.product_type.astype(int)

X.loc[X.product_gender=='Women','product_gender'] = '-1'
X.loc[X.product_gender=='Unisex','product_gender'] = '0'
X.loc[X.product_gender=='Men','product_gender'] = '1'
X.product_gender = X.product_gender.astype(int)

# 变换标签
X['y'] = np.log(X['target']+1)
print(X.head())

# 整合测试集
Z = test.copy()
Z = Z.merge(navigation_stats, on = 'sku_hash', how = 'left')
Z = Z.merge(sales_stats, on = 'sku_hash', how = 'left')
Z = Z.merge(traffic_source_views, on = 'sku_hash', how = 'left')
Z = Z.merge(type_sales, on = 'sku_hash', how = 'left')
Z = Z.merge(zone_sales, on = 'sku_hash', how = 'left')

Z.loc[Z.product_type=='Accessories','product_type'] = '0'
Z.loc[Z.product_type=='Leather Goods','product_type'] = '1'
Z.product_type = Z.product_type.astype(int)

Z.loc[Z.product_gender=='Women','product_gender'] = '-1'
Z.loc[Z.product_gender=='Unisex','product_gender'] = '0'
Z.loc[Z.product_gender=='Men','product_gender'] = '1'
Z.product_gender = Z.product_gender.astype(int)

features = ['product_type', 'product_gender',
            'page_views', 'sales_quantity',
            'TotalBuzzPost', 'TotalBuzz', 'NetSentiment', 'PositiveSentiment', 'NegativeSentiment', 'Impressions',
            'fr_FR_price',
            'macro_function_count', 'function_count', 'sub_function_count', 'model_count', 'aesthetic_sub_line_count', 'macro_material_count', 'color_count',
            'page_views_nav1', 'page_views_nav2', 'page_views_nav3', 'page_views_nav4', 'page_views_nav5', 'page_views_nav6',
            'sales_quantity_type1', 'sales_quantity_type2',
            'sales_quantity_zone1','sales_quantity_zone2','sales_quantity_zone3', 'sales_quantity_zone4','sales_quantity_zone5',
            'mean_target',]

# 交叉验证
# 选取某一个月的情况

def train_test_split(tr, te, mo, feats, num_folds):
    Xtrain = []
    ytrain = []
    dtrain = []
    Xval = []
    yval = []
    dval = []

    for i in range(num_folds):
        Xtrain.append(tr.loc[(tr.month == mo) & (tr.idx != i), feats].values)
        ytrain.append(tr.loc[(tr.month == mo) & (tr.idx != i), 'y'].values)
        dtrain.append(xgb.DMatrix(Xtrain[i], ytrain[i]))

        Xval.append(tr.loc[(tr.month == mo) & (tr.idx == i), feats].values)
        yval.append(tr.loc[(tr.month == mo) & (tr.idx == i), 'y'].values)
        dval.append(xgb.DMatrix(Xval[i], yval[i]))

    Xtest = te.loc[(te.month == mo), feats].values
    dtest = xgb.DMatrix(Xtest)

    return dtrain, dval, dtest

# Xgboost参数
param = {}
param['objective'] = 'reg:linear'
param['eval_metric'] =  'rmse'
param['booster'] = 'gbtree'
param['eta'] = 0.025
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['num_parallel_tree'] = 3
param['min_child_weight'] = 25
param['gamma'] = 5
param['max_depth'] =  3
param['silent'] = 1

# 第一个月的情况
dtrain, dval, dtest = train_test_split(tr=X, te=Z, mo=1, feats=features, num_folds=5)

model_m1 = []
for i in range(5):
    model_m1.append(
        xgb.train(param,
                  dtrain[i],
                  50000,
                  [(dtrain[i], 'train'), (dval[i], 'eval')],
                  early_stopping_rounds=20,
                  verbose_eval=False
                  )
    )

oof_m1 = []
oof_test_m1 = []
for i in range(5):
    oof_m1.append(model_m1[i].predict(dval[i]))
    oof_test_m1.append(model_m1[i].predict(dtest))

test_m1 = np.mean(oof_test_m1, axis=0)

# 例子用法
m1={}
for i in range(5):
    m1 = {**m1,**dict(zip(X.loc[(X.month==1)&(X.idx==i),'sku_hash'],oof_m1[i]))}

my_dict = {'dota':123,'lol':456}
my_dict = {**my_dict,**{'superman':789}}
print(my_dict)

print(m1)

m1 = {**m1,**dict(zip(Z.loc[(Z.month==1),'sku_hash'],test_m1))}

print(m1)

# 第二个月情况
dtrain2, dval2, dtest2 = train_test_split(tr=X2, te=Z2, mo=2, feats=features2, num_folds=5)

model_m2 = []

for i in range(5):
    model_m2.append(
        xgb.train(
            param,
            dtrain2[i],
            50000,
            [(dtrain2[i], 'train'), (dval2[i], 'eval')],
            early_stopping_rounds=200,
            verbose_eval=False)
    )

# run predictions for the 2 month

oof_m2 = []
oof_test_m2 = []
for i in range(5):
    oof_m2.append(model_m2[i].predict(dval2[i]))
    oof_test_m2.append(model_m2[i].predict(dtest2))

test_m2 = np.mean(oof_test_m2, axis=0)

m2 = {}
for i in range(5):
    m2 = {**m2, **dict(zip(X.loc[(X.month == 2) & (X.idx == i), 'sku_hash'], oof_m2[i]))}

m2 = {**m2, **dict(zip(Z.loc[(Z.month == 2), 'sku_hash'], test_m2))}

oof_m2 = pd.DataFrame.from_dict(m2, orient='index').reset_index()
oof_m2.columns = ['sku_hash', 'oof_m2']

X3 = pd.merge(X2.copy(), oof_m2, on='sku_hash')
Z3 = pd.merge(Z2.copy(), oof_m2, on='sku_hash')
features3 = features2 + ['oof_m2']

# 第三个月情况
dtrain3, dval3, dtest3 = train_test_split(tr=X3, te=Z3, mo=3, feats=features3, num_folds=5)

model_m3 = []

for i in range(5):
    model_m3.append(
        xgb.train(
            param,
            dtrain3[i],
            50000,
            [(dtrain3[i], 'train'), (dval3[i], 'eval')],
            early_stopping_rounds=200,
            verbose_eval=False)
    )

# run predictions for the 3 month

oof_m3 = []
oof_test_m3 = []
for i in range(5):
    oof_m3.append(model_m3[i].predict(dval3[i]))
    oof_test_m3.append(model_m3[i].predict(dtest3))

test_m3 = np.mean(oof_test_m3, axis=0)

m3 = {}
for i in range(5):
    m3 = {**m3, **dict(zip(X.loc[(X.month == 3) & (X.idx == i), 'sku_hash'], oof_m3[i]))}

m3 = {**m3, **dict(zip(Z.loc[(Z.month == 3), 'sku_hash'], test_m3))}

oof_m3 = pd.DataFrame.from_dict(m3, orient='index').reset_index()
oof_m3.columns = ['sku_hash', 'oof_m3']

X3 = pd.merge(X3.copy(), oof_m3, on='sku_hash')
Z3 = pd.merge(Z3.copy(), oof_m3, on='sku_hash')

# 方便评估，设定一个

Z3['target'] = 0
Z3.loc[Z3.month == 1, 'target'] = Z3.loc[Z3.month == 1, 'oof_m1']
Z3.loc[Z3.month == 2, 'target'] = Z3.loc[Z3.month == 2, 'oof_m2']
Z3.loc[Z3.month == 3, 'target'] = Z3.loc[Z3.month == 3, 'oof_m3']

X3['pred_target'] = 0
X3.loc[X3.month == 1, 'pred_target'] = X3.loc[X3.month == 1, 'oof_m1']
X3.loc[X3.month == 2, 'pred_target'] = X3.loc[X3.month == 2, 'oof_m2']
X3.loc[X3.month == 3, 'pred_target'] = X3.loc[X3.month == 3, 'oof_m3']

# 评估结果
print(f"month1: {np.sqrt(np.mean((X3.loc[X3.month==1,'y'] - X3.loc[X3.month==1,'pred_target'])**2))}")
print(f"month2: {np.sqrt(np.mean((X3.loc[X3.month==2,'y'] - X3.loc[X3.month==2,'pred_target'])**2))}")
print(f"month3: {np.sqrt(np.mean((X3.loc[X3.month==3,'y'] - X3.loc[X3.month==3,'pred_target'])**2))}")
print(f"overall: {np.sqrt(np.mean((X3['y'] - X3['pred_target'])**2))}")

# 生成结果
Z3['target'] = np.exp(Z3.target)-1
final_sub = Z3[['ID','target']]
final_sub.to_csv(os.path.join(base_path,'silly-raddar-sub4.csv'),index=None)






