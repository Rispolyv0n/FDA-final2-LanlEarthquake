from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

import logging

class StackModel():
    def __init__(self, baseModelList, clfModel):
        logging.info('In stacking model - Initializaing...')
        self.baseModelList = baseModelList
        self.clfModel = clfModel
        self.n_fold = len(baseModelList)

    def run(self, x_train, y_train, x_test):
        # --- level 1 ---
        # kfold
        logging.info('In stacking model - Training base models...')
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)

        kf = KFold(n_splits = self.n_fold)

        # train & record

        base_model_out_train = pd.DataFrame(index=range(x_train.shape[0]), dtype=np.float64, columns=range(self.n_fold))
        base_model_out_test_list = [ pd.DataFrame(index=range(x_test.shape[0]), dtype=np.float64, columns=range(self.n_fold)) for x in range(self.n_fold) ]

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            logging.info('In stacking model - In K-Fold %d.' % i)
            print("TRAIN:", train_index, "TEST:", test_index)

            oof_train_x, oof_test_x = x_train.iloc[train_index], x_train.iloc[test_index]
            oof_train_y, oof_test_y = y_train.iloc[train_index], y_train.iloc[test_index]
            print(oof_train_x.shape)
            print(oof_test_x.shape)
            print(oof_train_y.shape)
            print(oof_test_y.shape)
            
            for j, model in enumerate(self.baseModelList):
                model.fit(oof_train_x, oof_train_y.values.ravel())
                base_model_out_train.iloc[test_index, j] = model.predict(oof_test_x)
                base_model_out_test_list[j].iloc[:, i] = model.predict(x_test)
                print(mean_absolute_error(oof_test_y, base_model_out_train.iloc[test_index, j])) # mae

        logging.info('In stacking model - Finish training base models.')
        print(base_model_out_train.shape)
        print(base_model_out_train.head())
        base_model_out_mean = pd.DataFrame(index=range(x_test.shape[0]), dtype=np.float64, columns=range(self.n_fold))
        for i, df in enumerate(base_model_out_test_list):
            base_model_out_mean.iloc[:, i] = df.mean(axis=1)
        print(base_model_out_mean.shape)
        print(base_model_out_mean.head())
        # print(base_model_out_mean.head(x_test.shape[0]))

        # --- level 2 ---
        # train
        logging.info('In stacking model - Training level 2 model...')
        print(base_model_out_train.shape)
        print(base_model_out_train.head())
        self.clfModel.fit(base_model_out_train, y_train.values.ravel())

        final_out_train = self.clfModel.predict(base_model_out_train)
        print(final_out_train.shape)
        print(mean_absolute_error(y_train, final_out_train))
        
        test_final_out = self.clfModel.predict(base_model_out_mean)
        print(base_model_out_mean.shape)
        print(base_model_out_mean.head())

        logging.info('In stacking model - Finish.')
        return test_final_out

        
