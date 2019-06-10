import pandas as pd
import numpy as np
from tqdm import tqdm

from os import listdir
from os.path import isfile, join
import datetime
import logging
import sys

import feature_extraction as FE
from model import stacking as ST

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


# Initialization

logging.basicConfig(level=logging.DEBUG,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')

logging.info('Initializing.')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:20,.60f}'.format)
pd.set_option('display.max_colwidth', -1)

train_data_path = './data/train.csv'
test_file_path = './data/test'
test_file_names = [f for f in listdir(test_file_path) if isfile(join(test_file_path, f))]
submission_data_path = './data/sample_submission.csv'

n_rows = 150000
read_data_row = n_rows * 4000
skip_data_row = 0


# Read data - training data

logging.info('Reading training data(incomplete).')

train_df = pd.read_csv(train_data_path, nrows=read_data_row, skiprows=skip_data_row)
print(train_df.shape)
# print(train_df.head(5))
# print('==========')
# print(train_df.tail(5))
logging.info('Reading training data done(incomplete).')

# Read data - testing data

logging.info('Reading testing data(incomplete).')
test_df_list = []
read_num = 1
cur_num = 0
for name in test_file_names:
    test_df_list.append( pd.read_csv(test_file_path + '/' + name) )
    cur_num += 1
    if(cur_num >= read_num):
        break
print(len(test_df_list))
logging.info('Reading testing data done(incomplete).')

# Read data - submission data

submission_df = pd.read_csv(submission_data_path)


# Feature extraction - training data

logging.info('Extracting features(training data)...')
segments = int(np.floor(train_df.shape[0] / n_rows))
# print(segments)
train_x = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

for seg_id in tqdm(range(segments)):
    seg = train_df.iloc[seg_id * n_rows : seg_id * n_rows + n_rows]
    FE.create_features(seg_id, seg, train_x)
    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

# check !
logging.info('Checking features:')
# print(train_x.head())
print(train_x.shape)
# print(train_y.head())
print(train_y.shape)


# Feature extraction - testing data

logging.info('Extracting features(testing data)...')
segments = len(test_df_list)
# print(segments)
test_x = pd.DataFrame(index=range(segments), dtype=np.float64)

for seg_id in tqdm(range(segments)):
    seg = test_df_list[seg_id]
    FE.create_features(seg_id, seg, test_x)

# check !
logging.info('Checking features:')
# print(test_x.head())
print(test_x.shape)


# Initialize models

clf_rf = RandomForestRegressor()
clf_tree = ExtraTreesRegressor()
clf_ada = AdaBoostRegressor()
clf_grad = GradientBoostingRegressor()
clf_svr = SVR()

base_model_list = [clf_rf, clf_tree, clf_ada, clf_grad, clf_svr]
clf_xgb = xgb.XGBRegressor()

m_stacking = ST.StackModel(baseModelList = base_model_list, clfModel = clf_xgb)
predict_y = m_stacking.run(x_train = train_x, y_train = train_y, x_test = train_x) # x_test = test_x
print(predict_y)
print(predict_y.shape)
print(mean_absolute_error(train_y, predict_y))



