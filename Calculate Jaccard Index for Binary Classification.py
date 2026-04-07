import numpy as np


def jaccard_index(y_true, y_pred):
    # Write your code here

    # jaccard is very similar to F1
    TruePositive = 0
    FalsePositive = 0
    FalseNegative = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TruePositive += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FalsePositive += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FalseNegative += 1

    # jaccard = TP / (TP + FP + FN)
    Jaccard = TruePositive / (TruePositive + FalsePositive + FalseNegative)

    return round(Jaccard, 3)
