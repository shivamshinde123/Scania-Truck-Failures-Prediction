import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
import pickle
from utility_function import Utility
from sklearn.metrics import ConfusionMatrixDisplay

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

plot_metrics_path = params['logging_folder_paths']['plot_metrics']

file_handler = logging.FileHandler(plot_metrics_path)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class CreateViz:

    def __init__(self) -> None:
        pass

    def plot_precision_recall_vs_threshold(self, precisions, recalls, thresholds):
        plt.figure(figsize=(12, 8))
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
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plots_folder = params['plots']['plots_folder']
        pr_name = params['plots']['pr_name']

        Utility().create_folder(plots_folder)
        plt.savefig(os.path.join(plots_folder, pr_name))

    def plot_confusion_matrix(self,confusion_matrix):
        ax= plt.subplot()
        sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 

        plots_folder = params['plots']['plots_folder']
        cm_name = params['plots']['cm_name']

        Utility().create_folder(plots_folder)
        plt.savefig(os.path.join(plots_folder, cm_name))

        
    def plots(self):

        metrics_folder = params['metrics_path']['metrics_folder']
        metrics_file = params['metrics_path']['metrics_file']

        metrics = json.loads(os.path.join(metrics_folder, metrics_file))

        confusion_matrix = metrics['confusion_matrix']
        precisions = metrics['precisions']
        recalls = metrics['recalls']
        thresholds = metrics['thresholds']

        self.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        self.plot_precision_vs_recall(precisions, recalls)
        self.plot_confusion_matrix(confusion_matrix)


if __name__ == "__main__":

    cv = CreateViz()
    cv.plots()