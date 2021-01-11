from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score, multilabel_confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def metric_results(pred_label, gt_label):
    # 计算一些评价标准
    result = {}
    label_classes = 206
    precision = 0
    accuracy = 0
    recall = 0
    F1_score = 0
    # print(gt_label)
    for i in range(label_classes):
        precision += precision_score(gt_label[:, i], pred_label[:, i], average='weighted')
        accuracy += accuracy_score(gt_label[:, i], pred_label[:, i])
        recall += recall_score(gt_label[:, i], pred_label[:, i], average='weighted')
        F1_score += f1_score(gt_label[:, i], pred_label[:, i], average='weighted')

    # cm = multilabel_confusion_matrix(gt_label, pred_label)
    # result.update({'confusion_matrix': cm})

    # precision = precision_score(gt_label, pred_label, average='weighted')
    result.update({'precision': precision/label_classes})

    # accuracy = accuracy_score(gt_label, pred_label)
    result.update({'accuracy': accuracy/label_classes})

    # recall = recall_score(gt_label, pred_label, average='weighted')
    result.update({'recall': recall/label_classes})  # recall和sensitivity计算方法一样

    # F1_score = f1_score(gt_label, pred_label, average='weighted')
    result.update({'F1_score': F1_score/label_classes})
    return result


def printMetricResults(myDict):
    for item in myDict:
        print(item, ":", '\n', myDict[item])
