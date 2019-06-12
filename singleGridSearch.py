import pandas as pd
import numpy as np
from tqdm import tqdm

from os import listdir
from os.path import isfile, join
import datetime
import logging
import sys

import feature_extraction as FE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


def grid_search_for_models(model, param, name, data):
    logging.info('Start grid searching - ' + name)
    gs = GridSearchCV(estimator=model,  
                        param_grid=param,
                        scoring='neg_mean_absolute_error',
                        cv=5)
    gs.fit(data.drop(['time_to_failure'], axis=1), data['time_to_failure'])

    logging.info('=====================================')
    print(gs.best_params_)
    print(gs.best_score_)
    return


# Initialization

logging.basicConfig(level=logging.DEBUG,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')

logging.info('Initializing.')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:20,.60f}'.format)
pd.set_option('display.max_colwidth', -1)

n_rows = 150000
n_data = 4190 # 4190
read_data_row = n_rows * n_data
skip_data_row = 0

train_data_path = './data/train.csv'

feature_file_path_train = './data/features-'+str(n_data)+'.csv'

read_feature_from_file = True


# Read feature data or original data

if(read_feature_from_file):
    if(not isfile(feature_file_path_train)):
        logging.ERROR('Cannot find the training feature file.')
        sys.exit(1)
    
    logging.info('Reading training feature data: %s.' % feature_file_path_train)
    feature_df = pd.read_csv(feature_file_path_train, sep='\t')
    feature_df.drop(['Unnamed: 0'], axis=1, inplace=True) # drop 'Unnamed index'
    # TODO - feature filtering
    print(feature_df.shape)

else:
    # Read data - training data

    logging.info('Reading training data(incomplete).')
    train_df = pd.read_csv(train_data_path, nrows=read_data_row, skiprows=skip_data_row)
    print(train_df.shape)

    # Feature extraction - training data

    logging.info('Extracting features(training data)...')
    segments = int(np.floor(train_df.shape[0] / n_rows))
    train_x = pd.DataFrame(index=range(segments), dtype=np.float64)
    train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

    for seg_id in tqdm(range(segments)):
        seg = train_df.iloc[seg_id * n_rows : seg_id * n_rows + n_rows]
        FE.create_features(seg_id, seg, train_x)
        train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

    # write to csv file
    logging.info('Writing features to csv(training data)...')
    feature_df = pd.concat([train_x, train_y], axis=1)
    feature_df.to_csv(feature_file_path_train, sep='\t', encoding='utf-8')

    # check
    logging.info('Checking features of training data:')
    print(train_x.shape)
    print(train_y.shape)



# Feature scaling

logging.info('Scaling features...')

column_name_list = list(feature_df.columns)
print(len(column_name_list))
column_name_list.remove('time_to_failure')
print(len(column_name_list))

feature_scaler = StandardScaler()
feature_df[column_name_list] = feature_scaler.fit_transform(feature_df[column_name_list])


# Initialize models

clf_ridg = Ridge(max_iter=5000)
clf_laso = Lasso(max_iter=5000)
clf_lala = LassoLars(max_iter=5000)
clf_enet = ElasticNet(max_iter=5000)

clf_xgbr = xgb.XGBRegressor()
clf_xgrf = xgb.XGBRFRegressor()

clf_rf = RandomForestRegressor()
clf_tree = ExtraTreesRegressor()
clf_ada = AdaBoostRegressor()
clf_grad = GradientBoostingRegressor()
clf_svr = SVR()

# base_model_name = ['Ridge', 'Lasso', 'LassoLars', 'ElasticNet', 'XgbReg', 'XgbRf', 'RandomForest', 'ExtraTree', 'AdaBoost', 'GradientBoosting', 'SVR']
# base_model_list = [clf_ridg, clf_laso, clf_lala, clf_enet, clf_xgbr, clf_xgrf, clf_rf, clf_tree, clf_ada, clf_grad, clf_svr]

param_ridg = {
    'alpha': [1, 10, 30, 100, 300, 1000],
    'tol': [0.00001, 0.0000001, 0.000000001, 0.00000000001],
    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
}

param_laso = {
    'alpha': [1, 10, 100, 1000],
    'tol': [0.00001, 0.0000001, 0.000000001],
}

param_lala = {
    'alpha': [1, 10, 100, 1000],
    'eps': [2.220446049250313e-16, 2.220446049250313e-10]
}

param_enet = {
    'alpha': [1, 10, 100],
    'tol': [0.00001, 0.0000001, 0.000000001],
    'l1_ratio': [0, 0.5, 1],
}

param_xgbr = {
    'max_depth': [3, 8, 13, 20],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'gamma': [0, 0.000001, 0.0001],
}

param_xgrf = {
    'max_depth': [3, 8, 13, 20],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'gamma': [0, 0.000001, 0.0001],
}


# grid search - model arguments

# ridge
# logging.info('Start grid searching...')
# gs_ridg = GridSearchCV(estimator=clf_ridg,  
#                      param_grid=param_ridg,
#                      scoring='neg_mean_absolute_error',
#                      cv=5)
# gs_ridg.fit(feature_df.drop(['time_to_failure'], axis=1), feature_df['time_to_failure'])

# logging.info('Finish grid searching.')
# print(gs_ridg.best_params_)
# print(gs_ridg.best_score_)

# grid_search_for_models(model=clf_ridg, param=param_ridg, name='Ridge', feature_df)

grid_search_for_models(model=clf_laso, param=param_laso, name='Lasso', feature_df)
grid_search_for_models(model=clf_lala, param=param_lala, name='LassoLars', feature_df)
grid_search_for_models(model=clf_enet, param=param_enet, name='ElasticNet', feature_df)
grid_search_for_models(model=clf_xgbr, param=param_xgbr, name='XGB Regression', feature_df)
grid_search_for_models(model=clf_xgrf, param=param_xgrf, name='XGB RF', feature_df)

# grid_search_for_models(model=clf_ridg, param=param_ridg, name='Ridge', feature_df)
# grid_search_for_models(model=clf_ridg, param=param_ridg, name='Ridge', feature_df)
# grid_search_for_models(model=clf_ridg, param=param_ridg, name='Ridge', feature_df)
# grid_search_for_models(model=clf_ridg, param=param_ridg, name='Ridge', feature_df)



