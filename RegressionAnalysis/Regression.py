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