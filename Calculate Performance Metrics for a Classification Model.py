def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    # Implement your code here

    TruePositive = TrueNegative = FalsePositive = FalseNegative = 0

    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            TruePositive += 1
        elif actual[i] == 1 and predicted[i] == 0:
            FalseNegative += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FalsePositive += 1
        else:
            TrueNegative += 1

    precision = TruePositive / (TruePositive + FalsePositive)
    recall = TruePositive / (TruePositive + FalseNegative)

    confusion_matrix = [[TruePositive, FalseNegative], [FalsePositive, TrueNegative]]
    accuracy = (TruePositive + TrueNegative) / (TruePositive + TrueNegative + FalsePositive + FalseNegative)
    f1 = 2 * precision * recall / (precision + recall)
    specificity = TrueNegative / (TrueNegative + FalsePositive)
    negativePredictive = TrueNegative / (TrueNegative + FalseNegative)

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
