import logging
import os
from utility_function import Utility

import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

feature_path = params['logging_folder_paths']['data']

file_handler = logging.FileHandler(feature_path)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class MakeDataset:

    def __init__(self) -> None:
        pass

    def load_and_save(self, url, filename):
        """This method is used to load the data from google drive and to save the loaded data

        Parameters
        -----------

        url: URL of the data
        Returns
        --------
        None
        """

        inital_url = params['data']['data_url_base']

        url = inital_url + url.split('/')[-2]

        data = pd.read_csv(url)

        raw_data_path = params['data']['raw_data']

        Utility().create_folder('Data')
        Utility().create_folder(os.path.join('Data', 'raw'))

        data.to_csv(os.path.join(raw_data_path, str(
            filename) + '.csv'), index=False)


if __name__ == "__main__":

    train_data_url = params['data']['train_data_url']
    test_data_url = params['data']['test_data_url']

    md = MakeDataset()
    logger.info('Loading of train data started.')
    md.load_and_save(train_data_url, 'train')
    logger.info(
        'Train data loading completed and data saved to the directory Data/raw')
    logger.info('Loading of test data started.')
    md.load_and_save(test_data_url, 'test')
    logger.info(
        'Test data loading completed and data saved to the directory Data/raw')
