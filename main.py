import pandas as pd
import numpy as np
from tqdm import tqdm

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from os import listdir
from os.path import isfile, join
import datetime
import logging
import sys

import feature_extraction as FE
from model import stacking as ST

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
feature_file_path_test = './data/features-test.csv'

plot_feature_imp_file_path = './plot/featImp-'
plot_model_corr_test_file_path = './plot/modelCorr-test.png'
plot_model_corr_train_file_path = './plot/modelCorr-train.png'

plot_feature_importance = True
plot_model_correlation = False
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
    # read_num = 5
    # cur_num = 0
    for name in test_file_names:
        test_df_list.append( pd.read_csv(test_file_path + '/' + name) )
        # cur_num += 1
        # if(cur_num >= read_num):
            # break
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
    test_x.to_csv(feature_file_path_test, sep='\t', encoding='utf-8')

    # check
    logging.info('Checking features of testing data:')
    print(test_x.shape)



# Feature scaling

logging.info('Scaling features...')

column_name_list = list(feature_df.columns)
print(len(column_name_list))
column_name_list.remove('time_to_failure')
print(len(column_name_list))

feature_scaler = StandardScaler()
feature_df[column_name_list] = feature_scaler.fit_transform(feature_df[column_name_list])
test_x = feature_scaler.transform(test_x)


# Initialize models

clf_line = LinearRegression()
clf_ridg = Ridge()
clf_laso = Lasso()
clf_lala = LassoLars()
clf_enet = ElasticNet()
clf_bxgb = xgb.XGBRegressor()

clf_rf = RandomForestRegressor()
clf_tree = ExtraTreesRegressor()
clf_ada = AdaBoostRegressor()
clf_grad = GradientBoostingRegressor()
clf_svr = SVR()

base_model_name = ['RandomForest', 'ExtraTree', 'AdaBoost', 'GradientBoosting', 'SVR']
base_model_list = [clf_rf, clf_tree, clf_ada, clf_grad, clf_svr]

# base_model_name = ['LinearReg', 'Ridge', 'Lasso', 'LassoLars', 'ElasticNet', 'Xgb', 'RandomForest', 'ExtraTree', 'AdaBoost', 'GradientBoosting', 'SVR']
# base_model_list = [clf_line, clf_ridg, clf_laso, clf_lala, clf_enet, clf_bxgb, clf_rf, clf_tree, clf_ada, clf_grad, clf_svr]

clf_xgb = xgb.XGBRegressor()

m_stacking = ST.StackModel(baseModelList = base_model_list, clfModel = clf_xgb)
predict_y = m_stacking.run(x_train = feature_df.drop(['time_to_failure'], axis=1), y_train = feature_df['time_to_failure'], x_test = feature_df.drop(['time_to_failure'], axis=1), get_feat_imp=plot_feature_importance ) # x_test = test_x
logging.info('Finish Training.')
print(predict_y)
print(predict_y.shape)
print(mean_absolute_error(feature_df['time_to_failure'], predict_y)) # prediction mae


# Plot feature importance

if(plot_feature_importance):
    logging.info('Plot feature importance...')
    feat_importance_list = m_stacking.get_feature_importance()
    feat_names = column_name_list

    for i,impor in enumerate(feat_importance_list):
        imp = impor
        imp,names = zip(*sorted(zip(imp,feat_names)))
        plt.figure(i)
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        print('=================')
        print(names)
        fig = plt.gcf()
        fig.set_size_inches(36.5, 44.5)
        # plt.show()
        fig.savefig(plot_feature_imp_file_path + base_model_name[i] + '.png')


if(plot_model_correlation):
    # Plot model correlation - test

    logging.info('Plot model correlation - test...')
    base_prediction = m_stacking.get_base_prediction()
    print(base_prediction.shape)

    base_prediction.columns = base_model_name
    corr = base_prediction.corr()

    sns.set()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, heatmap_ax = plt.subplots(figsize=(10,10))

    sns.heatmap(corr, cmap=cmap, vmax=1, center=0, square=True, linewidths=0, annot=True, ax=heatmap_ax)
    # plt.show()
    plt.savefig(plot_model_corr_test_file_path)


    # Plot model correlation - train

    logging.info('Plot model correlation - train...')
    base_out = m_stacking.get_base_out_train()
    print(base_out.shape)

    base_out.columns = base_model_name
    corr = base_out.corr()

    sns.set()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, heatmap_ax = plt.subplots(figsize=(10,10))

    sns.heatmap(corr, cmap=cmap, vmax=1, center=0, square=True, linewidths=0, annot=True, ax=heatmap_ax)
    # plt.show()
    plt.savefig(plot_model_corr_train_file_path)


# grid search - model arguments

