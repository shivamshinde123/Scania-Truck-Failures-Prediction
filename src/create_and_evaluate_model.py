import logging
import os
import json
import joblib
import pickle
from utility_function import Utility
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

create_model_path = params['logging_folder_paths']['model_creation']

file_handler = logging.FileHandler(create_model_path)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class CreateModel:

    def __init__(self) -> None:
        pass

    def make_model(self):

        # STAGE 1: Loading preprocessed data
        data_folder = params['data']['processed_data']
        data = pd.read_csv(os.path.join(data_folder, 'processed_train.csv'))

        # STAGE 2: Splitting the data into train data and validation data
        X = data.drop(columns=['class'], axis=1)
        y = data['class']

        random_state = params['base']['random_state']

        split_ratio = params['base']['split_ratio']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, random_state=random_state, test_size=split_ratio, stratify=y)

        # STAGE 3: Creating a model
        max_depth = params['model']['rfc']['max_depth']
        max_features = params['model']['rfc']['max_features']
        min_samples_split = params['model']['rfc']['min_samples_split']
        min_samples_leaf = params['model']['rfc']['min_samples_leaf']
        n_jobs = params['base']['n_jobs']
    
        rfc = RandomForestClassifier(max_depth=max_depth, max_features=max_features,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     n_jobs=n_jobs, random_state=random_state)

        # STAGE 4: Training a model
        model = rfc.fit(X_train, y_train)

        # STAGE 5: Making predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        # STAGE 6: Finding different metrics
        positive_decision_score = y_proba[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(
            y_val, positive_decision_score)

        cm = confusion_matrix(y_val, y_pred, normalize='all',labels=model.classes_)

        auc_roc_scr = roc_auc_score(y_val, positive_decision_score)

        ## STAGE 7: Saving the trained model as python pickle file
        model_foldername = params['model']['model_foldername']
        model_name = params['model']['model_name']

        Utility().create_folder(model_foldername)
        
        with open(os.path.join(model_foldername, model_name), 'wb') as f:
            pickle.dump(model, f)

        # STAGE 8: Saving the calculated metrics
        metrics_folder = params['metrics_path']['metrics_folder']
        metrics_file = params['metrics_path']['metrics_file']

        metrics = {
            'confusion_matrix': cm.tolist(),
            'precisions': precisions,
            'recalls': recalls,
            'thresholds': thresholds,
            'auc_roc_score': auc_roc_scr
        }

        Utility().create_folder(metrics_folder)

        with open(os.path.join(metrics_folder, metrics_file), 'w') as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    
    model_creation = CreateModel()
    model_creation.make_model()