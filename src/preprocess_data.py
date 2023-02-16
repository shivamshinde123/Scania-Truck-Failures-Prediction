
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

import logging
import os
import pandas as pd
import numpy as np
from utility_function import Utility


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

make_dataset_path = params['logging_folder_paths']['data']

file_handler = logging.FileHandler(make_dataset_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Remove_Useless_Features(BaseEstimator, TransformerMixin):
    
    def __init__(self, column_names_list):
        self.column_names_list = column_names_list
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X.drop(columns=self.column_names_list, axis=1, inplace=True)
        return X

class replace_na_string_with_numpy_na(BaseEstimator, TransformerMixin):

    def __init__(self): 
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        def miss(x):
            if x == 'na':
                return np.nan
            else:
                return x

        for feature in X.columns:
            X[feature] = X[feature].map(miss)

        return X

class Outlier_Remover(BaseEstimator, TransformerMixin):
    
    def __init__(self): 
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        quantiles = X.quantile(np.arange(0,1,0.25)).T
        quantiles = quantiles.rename(columns={0.25:'Q1', 0.50: 'Q2', 0.75:'Q3'})
        
        quantiles['IQR'] = quantiles['Q3'] - quantiles['Q1']
        quantiles['Lower_Limit'] = quantiles['Q1'] - 1.5*quantiles['IQR']
        quantiles['Upper_Limit'] = quantiles['Q3'] + 1.5*quantiles['IQR']
        
        for feature in X.columns:
            X[feature] = np.where((X[feature] < quantiles.loc[feature,'Lower_Limit']) | (X[feature] > quantiles.loc[feature,'Upper_Limit']) & (X[feature] is not np.nan), X[feature].median(), X[feature])
        
        return X


class Log_Transformer(BaseEstimator, TransformerMixin):
    
    def __init__(self): 
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for feature in X.columns:
                X[feature] = np.where(X[feature]==0,np.log(X[feature]+0.0002),np.log(X[feature]))
        return X


class Preprocess:

    def __init__(self):
        pass


    def preprocss(self, X, process_data_filename):

        ## STAGE 1: Creating preprocessing pipelines
        num_pipe = Pipeline(steps=[
                    ('outlier_removal',Outlier_Remover()),
                    ('replace_na',replace_na_string_with_numpy_na()),
                    ('replacing_num_missing_values',SimpleImputer(strategy='median', missing_values=np.nan)),
                    ('scaling',MinMaxScaler()),
                    ('log_transformation',Log_Transformer()),
                    ('remove_useless_feat',Remove_Useless_Features(['ab_000', 'ad_000', 'bk_000', 'bl_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 
                                    'bq_000', 'br_000', 'cf_000', 'cg_000', 'ch_000', 'co_000', 'cr_000', 'ct_000', 'cu_000', 'cv_000', 
                                    'cx_000', 'cy_000', 'cz_000', 'da_000', 'db_000', 'dc_000']))
                ])


        ## STAGE 2: Splitting the data into independent and dependent features
        target_col_name = params['base']['target_col_name']
        column_names = X.columns[1:]
        X, y = pd.DataFrame(X.drop(target_col_name, axis=1), columns=column_names), X[target_col_name]
        
        ## STAGE 3: Preprocessing the data using created pipelines
        processed_data = num_pipe.fit_transform(X)
        processed_data[target_col_name] = y

        ## STAGE 3: Saving the preprocess pipelines and process_data
        processed_data_path = params['data']['processed_data']
        processed_data.to_csv(os.path.join(processed_data_path, 'processed_' + str(process_data_filename)  + '.csv'), index=False)


        preprocess_pipe_folderpath = params['model']['preprocess_pipe_folderpath']
        preprocess_pipe_filename = params['model']['preprocess_pipe_filename']

        preprocess_pipe_path = os.path.join(preprocess_pipe_folderpath, preprocess_pipe_filename)
        joblib.dump(num_pipe, open(preprocess_pipe_path, 'wb'))



if __name__ == "__main__":

    p = Preprocess()
    raw_data_path = params['data']['raw_data']
    train = pd.read_csv(os.path.join(raw_data_path, 'train.csv'))
    p.preprocss(train, 'train')


        