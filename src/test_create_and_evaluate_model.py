

import json
import os
import pytest
import dill
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from src.utility_function import Utility


@pytest.fixture
def params():
    """This fixture is used to load the params.yaml file"""
    return Utility().read_params()


class ModelUnderfitting(Exception):
    """Raised when the ROC AUC score of the trained model is equal or less than 0.5"""
    pass


class ModelOverfitting(Exception):
    """Raised when the difference between the ROC AUC score of train data and test data is more than or equal to 0.25"""
    pass


def test_check_the_saved_ml_model(params):

    """This python test is used to check if the trained model is saved in desired directory or not."""

    model_foldername = params['model']['model_foldername']
    model_name = params['model']['model_name']

    assert os.path.exists(os.path.join(model_foldername, model_name))


def test_check_saved_evaluation_plots(params):

    """This python test is used to check if the visualizations of the evaluations are stored in the desired directory or not."""

    plots_folder = params['plots']['plots_folder']
    pr_thr_name = params['plots']['pr_thr_name']
    pr_name = params['plots']['pr_name']
    cm_name = params['plots']['cm_name']

    plot1_path = os.path.join(plots_folder, pr_thr_name)
    plot2_path = os.path.join(plots_folder, pr_name)
    plot3_path = os.path.join(plots_folder, cm_name)

    assert (os.path.exists(plot1_path) and os.path.exists(
        plot2_path) and os.path.exists(plot3_path))


def test_check_metrics(params):

    """This python test is used to check if the metrics calculated using trained machine learning model and the validation data are saved in desired 
            directory or not"""

    metrics_folder = params['metrics_path']['metrics_folder']
    metrics_file = params['metrics_path']['metrics_file']

    with open(os.path.join(metrics_folder, metrics_file), 'r') as f:
        metrics = json.load(f)

    roc_auc_score = metrics['auc_roc_score']
    precision = metrics['precision']
    recall = metrics['recall']

    assert (roc_auc_score > 0.5 and roc_auc_score <= 1)
    assert (precision > 0 and precision < 1)
    assert (recall > 0 and recall < 1)


def test_check_for_underfitting_and_overfitting(params):


    """This python test is used to check whether the trained model is overfitting the data or underfitting the data."""

    raw_data_path = params['data']['processed_data']
    train = pd.read_csv(os.path.join(raw_data_path, 'processed_train.csv'))

    target_col_name = params['base']['target_col_name']
    X, y = train.drop(target_col_name, axis=1), train[target_col_name]

    random_state = params['base']['random_state']

    split_ratio = params['base']['split_ratio']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, random_state=random_state, test_size=split_ratio, stratify=y)

    model_foldername = params['model']['model_foldername']
    model_name = params['model']['model_name']

    with open(os.path.join(model_foldername, model_name), 'rb') as f:
        model = dill.load(f)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_val = model.predict_proba(X_val)[:, 1]

    auc_roc_scr_train = roc_auc_score(y_train, y_pred_train)
    auc_roc_scr_val = roc_auc_score(y_val, y_pred_val)

    if (auc_roc_scr_train <= 0.5):
        raise ModelUnderfitting

    if (auc_roc_scr_train - auc_roc_scr_val > 0.25):
        raise ModelOverfitting
