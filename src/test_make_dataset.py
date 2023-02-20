
import pandas as pd
import pytest
from src.utility_function import Utility

@pytest.fixture
def params():
    return Utility().read_params()

def test_check_if_data_is_loaded_as_dataframe(params):

    ## getting data url from params.yaml file
    inital_url = params['data']['data_url_base']
    train_data_url = params['data']['train_data_url']

    url = inital_url + train_data_url.split('/')[-2]

    ## Reading the csv file
    data = pd.read_csv(url)

    assert isinstance(data, pd.DataFrame)

