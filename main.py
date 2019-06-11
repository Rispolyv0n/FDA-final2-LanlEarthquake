import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

n_rows = 150000
n_data = 4190
read_data_row = n_rows * n_data
skip_data_row = 0

train_data_path = './data/train.csv'
test_file_path = './data/test'
test_file_names = [f for f in listdir(test_file_path) if isfile(join(test_file_path, f))]
submission_data_path = './data/sample_submission.csv'
feature_file_path_train = './data/features-'+str(n_data)+'.csv'
feature_file_path_test = './data/features-test.csv'

plot_feature_importance = True
read_feature_from_file = False

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
    print(feature_df.shape)

    logging.info('Reading testing feature data: %s.' % feature_file_path_test)
    test_df = pd.read_csv(feature_file_path_test, sep='\t')
    fest_df.drop(['Unnamed: 0'], axis=1, inplace=True) # drop 'Unnamed index'
    print(test_df.shape)

else:
    # Read data - training data

    logging.info('Reading training data(incomplete).')

    train_df = pd.read_csv(train_data_path, nrows=read_data_row, skiprows=skip_data_row)
    print(train_df.shape)
    # print(train_df.head(5))
    # print('==========')
    # print(train_df.tail(5))

    # Read data - testing data

    logging.info('Reading testing data(incomplete).')
    test_df_list = []
    # read_num = 5
    # cur_num = 0
    for name in test_file_names:
        test_df_list.append( pd.read_csv(test_file_path + '/' + name) )
        # cur_num += 1
        # if(cur_num >= read_num):
        #     break
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
    feature_df.to_csv(feature_file_path_train, sep='\t', encoding='utf-8')

    # check !
    logging.info('Checking features of training data:')
    # print(train_x.head())
    print(train_x.shape)
    # print(train_y.head())
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
    test_x.to_csv(feature_file_path_test, sep='\t', encoding='utf-8')

    # check !
    logging.info('Checking features of testing data:')
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
# predict_y = m_stacking.run(x_train = train_x, y_train = train_y, x_test = train_x) # x_test = test_x
predict_y = m_stacking.run(x_train = feature_df.drop(['time_to_failure'], axis=1), y_train = feature_df['time_to_failure'], x_test = feature_df.drop(['time_to_failure'], axis=1) ) # x_test = test_x
print(predict_y)
print(predict_y.shape)
print(mean_absolute_error(feature_df['time_to_failure'], predict_y))


# plot feature importance

if(plot_feature_importance):
    logging.info('Plot feature importance...')
    feat_importance_list = m_stacking.get_feature_importance()
    feat_names = list(train_x.columns)

    for i,impor in enumerate(feat_importance_list):
        imp = impor
        imp,names = zip(*sorted(zip(imp,feat_names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        print('=================')
        print(names)
        fig = plt.gcf()
        fig.set_size_inches(36.5, 44.5)
        # plt.show()
        fig.savefig('./plot/featImp_'+str(i)+'.png')

# grid search - model arguments

