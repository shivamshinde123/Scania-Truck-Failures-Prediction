
import pytest
import pandas as pd
import os
from src.utility_function import Utility

@pytest.fixture
def params():
    return Utility().read_params()


def test_check_input_shape(params):

    """This python test is used to check the shape of input raw data."""

    raw_data_path = params['data']['raw_data']
    train = pd.read_csv(os.path.join(raw_data_path, 'train.csv'))

    assert train.shape == (60000, 171)


def test_check_output_shape(params):

    """This python test is used to check the shape of data after preprocessing performed."""

    processed_data_path = params['data']['processed_data']
    processed_data = pd.read_csv(os.path.join(processed_data_path, 'processed_train.csv'))

    assert processed_data.shape == (60000, 147)


def test_check_saved_preprocess_pipelines(params):

    """This python test is used to check if the trained preprocess pipeline saved in desired directory or not."""

    preprocess_pipe_folderpath = params['model']['preprocess_pipe_folderpath']
    preprocess_pipe_filename = params['model']['preprocess_pipe_filename']
    preprocess_pipe_path = os.path.join(
                preprocess_pipe_folderpath, preprocess_pipe_filename)

    assert os.path.exists(preprocess_pipe_path)


def test_check_saved_label_encoder(params):

    """This python test is used to check if the trained label encoder is saved in the desired directory or not."""

    preprocess_pipe_folderpath = params['model']['preprocess_pipe_folderpath']
    label_encoder_filename = params['model']['label_encoder_filename']

    le_file_path = os.path.join(
        preprocess_pipe_folderpath, label_encoder_filename)

    assert os.path.exists(le_file_path)


