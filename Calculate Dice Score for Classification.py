import numpy as np


def dice_score(y_true, y_pred):
    # Write your code here
    Intersection = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            Intersection += 1
    num_true_positives = 0
    num_predicted_positives = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            num_true_positives += 1
        if y_pred[i] == 1:
            num_predicted_positives += 1
    if num_true_positives == 0:
        return 0.0

    diceScore = 2 * Intersection / (num_true_positives + num_predicted_positives)

    return round(diceScore, 3)
