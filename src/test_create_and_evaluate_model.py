

import json
import os
import pytest

from src.utility_function import Utility


@pytest.fixture
def params():
    return Utility().read_params()


def test_check_the_saved_ml_model(params):

    model_foldername = params['model']['model_foldername']
    model_name = params['model']['model_name']

    assert os.path.exists(os.path.join(model_foldername, model_name))


def test_check_saved_evaluation_plots(params):

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
