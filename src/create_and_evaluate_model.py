import logging
import os
import json
import dill
import numpy as np
import pickle
from utility_function import Utility
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, precision_score, recall_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    def plot_precision_recall_vs_threshold(self, precisions, recalls, thresholds):
        plt.figure(figsize=(12, 8))
        sns.set_style('darkgrid')
        plt.plot(thresholds, precisions[:-1], 'b', label="Precision")
        plt.plot(thresholds, recalls[:-1], 'g', label="Recall")
        plt.axhline(y=0.95, label="high recall: 0.95",
                    color='r', linestyle="--")
        plt.legend(loc='best')
        plt.xlabel('Threshold')

        plots_folder = params['plots']['plots_folder']
        pr_thr_name = params['plots']['pr_thr_name']

        Utility().create_folder(plots_folder)
        plt.savefig(os.path.join(plots_folder, pr_thr_name))

    def plot_precision_vs_recall(self, precisions, recalls):
        plt.figure(figsize=(12, 8))
        sns.set_style('darkgrid')
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plots_folder = params['plots']['plots_folder']
        pr_name = params['plots']['pr_name']

        Utility().create_folder(plots_folder)
        plt.savefig(os.path.join(plots_folder, pr_name))

    def plot_confusion_matrix(self, confusion_matrix):
        ax = plt.subplot()
        sns.set_style('darkgrid')
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix),
                    annot=True, fmt='.2%', ax=ax, cmap='Blues')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

        plots_folder = params['plots']['plots_folder']
        cm_name = params['plots']['cm_name']

        Utility().create_folder(plots_folder)
        plt.savefig(os.path.join(plots_folder, cm_name))

    def make_model(self):

        logger.info('Model creation step started.')
        # STAGE 1: Loading preprocessed data
        data_folder = params['data']['processed_data']
        data = pd.read_csv(os.path.join(data_folder, 'processed_train.csv'))
        logger.info('Processed data loaded.')

        # STAGE 2: Splitting the data into train data and validation data
        X = data.drop(columns=['class'], axis=1)
        y = data['class']
        logger.info(
            'Processed data splitted into independent features and dependent features')

        random_state = params['base']['random_state']

        split_ratio = params['base']['split_ratio']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, random_state=random_state, test_size=split_ratio, stratify=y)

        logger.info(
            'Processed data splitted into the data for training and validation.')

        # STAGE 3: Creating a model
        max_depth = params['model']['rfc']['max_depth']
        max_features = params['model']['rfc']['max_features']
        min_samples_split = params['model']['rfc']['min_samples_split']
        min_samples_leaf = params['model']['rfc']['min_samples_leaf']
        n_jobs = params['base']['n_jobs']

        rfc = RandomForestClassifier(max_depth=max_depth, max_features=max_features,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     n_jobs=n_jobs, random_state=random_state)

        logger.info('Created a model object.')

        # STAGE 4: Training a model
        model = rfc.fit(X_train, y_train)
        logger.info(
            'Model successfully trained on the processed training data.')

        # STAGE 5: Making predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        logger.info('Predictions made using the trained model.')

        # STAGE 6: Finding different metrics
        positive_decision_score = y_proba[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(
            y_val, positive_decision_score)

        cm = confusion_matrix(
            y_val, y_pred, normalize='all', labels=model.classes_)

        auc_roc_scr = roc_auc_score(y_val, positive_decision_score)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        clf_report = classification_report(y_val, y_pred, output_dict=True)
        clf_report = pd.DataFrame(clf_report).transpose()

        logger.info('Different metrics were calculated using trained model.')
        
        # STAGE 7: Saving the trained model as python pickle file
        model_foldername = params['model']['model_foldername']
        model_name = params['model']['model_name']

        Utility().create_folder(model_foldername)

        with open(os.path.join(model_foldername, model_name), 'wb') as f:
            dill.dump(model, f)

        logger.info('Trained model saved into python pickle file.')

        # STAGE 8: Saving the calculated metrics
        metrics_folder = params['metrics_path']['metrics_folder']
        metrics_file = params['metrics_path']['metrics_file']
        clf_report_path = params['metrics_path']['clf_report_filename']

        metrics = {
            'auc_roc_score': auc_roc_scr,
            'precision': precision,
            'recall': recall
        }

        Utility().create_folder(metrics_folder)

        with open(os.path.join(metrics_folder, metrics_file), 'w') as f:
            json.dump(metrics, f, indent=4)

        clf_report.to_csv(os.path.join(metrics_folder, clf_report_path))
        logger.info('Calculated metrics saved to the json and csv file.')

        # STAGE 9: Plotting metrics
        self.plot_precision_recall_vs_threshold(
            precisions, recalls, thresholds)
        self.plot_precision_vs_recall(precisions, recalls)
        self.plot_confusion_matrix(cm)

        logger.info(
            'Calculated metrics plotted for better understanding of model performance.')
        logger.info('Model creation step successfully completed.')


if __name__ == "__main__":

    model_creation = CreateModel()
    model_creation.make_model()
