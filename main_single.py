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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
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

n_rows = 150000
n_data = 4190 # 4190
read_data_row = n_rows * n_data
skip_data_row = 0

train_data_path = './data/train.csv'
test_file_path = './data/test'
test_file_names = [f for f in listdir(test_file_path) if isfile(join(test_file_path, f))]
submission_data_path = './data/sample_submission.csv'

feature_file_path_train = './data/features-'+str(n_data)+'.csv'
feature_file_path_test = './data/features-test-withId.csv'

read_feature_from_file = True


# Read feature data or original data

if(read_feature_from_file):
    if(not isfile(feature_file_path_train)):
        logging.ERROR('Cannot find the training feature file.')
        sys.exit(1)
    if(not isfile(feature_file_path_test)):
        logging.ERROR('Cannot find the testing feature file.')
        sys.exit(1)
    
    logging.info('Reading training feature data: %s.' % feature_file_path_train)
    feature_df = pd.read_csv(feature_file_path_train, sep='\t')
    feature_df.drop(['Unnamed: 0'], axis=1, inplace=True) # drop 'Unnamed index'
    # TODO - feature filtering
    print(feature_df.shape)

    logging.info('Reading testing feature data: %s.' % feature_file_path_test)
    test_x = pd.read_csv(feature_file_path_test, sep='\t')
    test_x.drop(['Unnamed: 0'], axis=1, inplace=True) # drop 'Unnamed index'
    # TODO - feature filtering
    print(test_x.shape)

else:
    # Read data - training data

    logging.info('Reading training data(incomplete).')
    train_df = pd.read_csv(train_data_path, nrows=read_data_row, skiprows=skip_data_row)
    print(train_df.shape)

    # Read data - testing data

    logging.info('Reading testing data(incomplete).')
    test_df_list = []
    test_name_list = []
    for name in test_file_names:
        test_df_list.append( pd.read_csv(test_file_path + '/' + name) )
        ind = name.find('.')
        test_name_list.append( name[:ind] )
    print(len(test_df_list))


    # Read data - submission data

    submission_df = pd.read_csv(submission_data_path)


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
    feature_df.to_csv(feature_file_path_train, sep='\t', encoding='utf-8', index=False)

    # check
    logging.info('Checking features of training data:')
    print(train_x.shape)
    print(train_y.shape)


    # Feature extraction - testing data

    logging.info('Extracting features(testing data)...')
    segments = len(test_df_list)
    test_x = pd.DataFrame(index=range(segments), dtype=np.float64)

    for seg_id in tqdm(range(segments)):
        seg = test_df_list[seg_id]
        FE.create_features(seg_id, seg, test_x)

    # write to csv file
    logging.info('Writing features to csv(testing data)...')
    test_x['seg_id'] = test_name_list
    test_x.to_csv(feature_file_path_test, sep='\t', encoding='utf-8', index=False)

    # check
    logging.info('Checking features of testing data:')
    print(test_x.shape)


# Feature scaling

logging.info('Scaling features...')

feat_column_name_list = list(feature_df.columns)
print(len(feat_column_name_list))
feat_column_name_list.remove('time_to_failure')
print(len(feat_column_name_list))

test_column_name_list = list(test_x.columns)
print(len(test_column_name_list))
test_column_name_list.remove('seg_id')
print(len(test_column_name_list))

feature_scaler = StandardScaler()
feature_df[feat_column_name_list] = feature_scaler.fit_transform(feature_df[feat_column_name_list])
test_x[test_column_name_list] = feature_scaler.transform(test_x[test_column_name_list])


# Initialize models

clf_line = LinearRegression()
clf_ridg = Ridge(alpha=300, tol=1e-05, solver='sparse_cg', max_iter=5000)
clf_laso = Lasso(alpha=0.1, tol=1e-05, max_iter=5000)
clf_lala = LassoLars(alpha=0.001, max_iter=5000)
clf_enet = ElasticNet(alpha=0.1, tol=0.001, l1_ratio=0.2, max_iter=5000)

clf_xgbr = xgb.XGBRegressor() # not yet
clf_xgrf = xgb.XGBRFRegressor() # not yet

clf_rf = RandomForestRegressor(criterion='mae', max_features='sqrt', n_estimators=200, max_depth=10)
clf_tree = ExtraTreesRegressor(criterion='mae', max_features='sqrt', n_estimators=200, max_depth=10)
clf_ada = AdaBoostRegressor(n_estimators=3, loss='linear')
clf_grad = GradientBoostingRegressor() # not yet
clf_svr = SVR(kernel='rbf', C=0.1)

base_model_name = ['LinearReg', 'Ridge', 'Lasso', 'LassoLars', 'ElasticNet', 'XgbReg', 'XgbRandomForest', 'RandomForest', 'ExtraTree', 'AdaBoost', 'GradientBoosting', 'SVR']
base_model_list = [clf_line, clf_ridg, clf_laso, clf_lala, clf_enet, clf_xgbr, clf_xgrf, clf_rf, clf_tree, clf_ada, clf_grad, clf_svr]


# Run models

for i, model in enumerate(base_model_list):
    logging.info('Training model - ' + base_model_name[i] + '...')
    model.fit(feature_df.drop(['time_to_failure'], axis=1), feature_df['time_to_failure'])
    predict_train = model.predict(feature_df.drop(['time_to_failure'], axis=1))
    print(mean_absolute_error(feature_df['time_to_failure'], predict_train)) # prediction_train mae
    predict_test = model.predict(test_x.drop(['seg_id'], axis=1))
    # write prediction csv
    result_df = pd.DataFrame(index=range(predict_test.shape[0]), dtype=np.float64)
    result_df['seg_id'] = test_x['seg_id']
    result_df['time_to_failure'] = predict_test
    submission_file_name = './data/output/singleModel-' + base_model_name[i] + '.csv'
    print(result_df.shape)
    print(result_df.head())
    result_df.to_csv(submission_file_name, sep=',', encoding='utf-8', index=False)
    break
    

