
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import dill
import logging
import os
import pandas as pd
import numpy as np
from utility_function import Utility


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

make_dataset_path = params['logging_folder_paths']['features']

file_handler = logging.FileHandler(make_dataset_path)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# class Remove_Useless_Features(BaseEstimator, TransformerMixin):

#     def __init__(self, column_names_list):
#         self.column_names_list = column_names_list

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X.drop(columns=self.column_names_list, axis=1, inplace=True)
#         return X

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

# class Outlier_Remover(BaseEstimator, TransformerMixin):

#     def __init__(self,columns):
#         self.columns = columns

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         quantiles = X.quantile(np.arange(0,1,0.25)).T
#         quantiles = quantiles.rename(columns={0.25:'Q1', 0.50: 'Q2', 0.75:'Q3'})

#         quantiles['IQR'] = quantiles['Q3'] - quantiles['Q1']
#         quantiles['Lower_Limit'] = quantiles['Q1'] - 1.5*quantiles['IQR']
#         quantiles['Upper_Limit'] = quantiles['Q3'] + 1.5*quantiles['IQR']

#         for feature in self.columns:
#             X[feature] = np.where((X[feature] < quantiles.loc[feature,'Lower_Limit']) | (X[feature] > quantiles.loc[feature,'Upper_Limit']) & (X[feature] is not np.nan), X[feature].median(), X[feature])

#         return X


# class Log_Transformer(BaseEstimator, TransformerMixin):

#     def __init__(self, columns):
#         self.columns = columns

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         for feature in self.columns:
#                 X[feature] = np.where(X[feature]==0,np.log(X[feature]+0.0002),np.log(X[feature]))
#         return X


class Preprocess:

    def __init__(self):
        pass

    def preprocss(self, X, process_data_filename):
        """This method is used to preprocess the input data.

        Parameters
        -----------

        X: Input data
        process_data_filename: keyword used in the name of processed data file

        Returns
        --------
        None
        """

        try:
            # STAGE 1: Removing features having more than 20% of missing values
            logger.info('Preprocessing stage started.')

            X.drop(columns=['ab_000', 'ad_000', 'bk_000', 'bl_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000', 'br_000', 'cf_000', 'cg_000', 'ch_000', 'co_000', 'cr_000',
                            'ct_000', 'cu_000', 'cv_000', 'cx_000', 'cy_000', 'cz_000', 'da_000', 'db_000', 'dc_000'],
                axis=1, inplace=True)

            logger.info(
                "Columns having more than 20% of missing values removed from the data.")

            # STAGE 2: Creating preprocessing pipelines
            num_pipe = Pipeline(steps=[
                ('replace_na', replace_na_string_with_numpy_na()),
                ('replacing_num_missing_values', SimpleImputer(
                    strategy='median', missing_values=np.nan)),
                ('scaling', MinMaxScaler()),
            ])

            # STAGE 3: Splitting the data into independent and dependent features
            target_col_name = params['base']['target_col_name']
            X, y = X.drop(target_col_name, axis=1), X[target_col_name]
            logger.info(
                'Splitted the input data into two parts: independent features and dependent feature.')

            # STAGE 4: Preprocessing the data using created pipelines
            processed_data = num_pipe.fit_transform(X)
            logger.info('Missing values imputation and min-max scaling completed.')
            processed_data = pd.DataFrame(processed_data)

            ## encoding the target column using label encoder
            le = LabelEncoder()
            processed_data[target_col_name] = le.fit_transform(y)

            # STAGE 5: Saving the preprocess pipelines and process_data
            ## Creating a folder to store the processed data
            Utility().create_folder(os.path.join('Data', 'processed'))

            ## Saving the processed data
            processed_data_path = params['data']['processed_data']
            processed_data.to_csv(os.path.join(
                processed_data_path, 'processed_' + str(process_data_filename) + '.csv'), index=False)
            logger.info('Processed data saved to the directory Data/processed.')

            ## Saving the preprocess pipeline
            preprocess_pipe_folderpath = params['model']['preprocess_pipe_folderpath']
            preprocess_pipe_filename = params['model']['preprocess_pipe_filename']

            Utility().create_folder(preprocess_pipe_folderpath)

            preprocess_pipe_path = os.path.join(
                preprocess_pipe_folderpath, preprocess_pipe_filename)
            
            with open(preprocess_pipe_path, 'wb') as f:
                dill.dump(num_pipe, f)

            logger.info(
                'Saved the fitted preprocess transformer in the python pickle file.')

            ## Saving the label encoder
            label_encoder_filename = params['model']['label_encoder_filename']

            le_file_path = os.path.join(
                preprocess_pipe_folderpath, label_encoder_filename)

            with open(le_file_path, 'wb') as f:
                dill.dump(le, f)

            logger.info(
                'Saved the fitted label encoder transformer in the python pickle file.')

            logger.info('Preprocessing stage completed.')
        
        except Exception as e:
            raise e


if __name__ == "__main__":

    p = Preprocess()
    raw_data_path = params['data']['raw_data']
    train = pd.read_csv(os.path.join(raw_data_path, 'train.csv'))
    p.preprocss(train, 'train')
