import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def forward(self, X, y):

        self.count_1 = np.sum(y == 1)
        self.count_0 = np.sum(y == 0)
        self.length_y = len(y)

        rows_1 = X[y == 1]
        rows_0 = X[y == 0]

        probs_1 = []
        probs_0 = []

        for i in range(X.shape[1]):
            cols_1 = rows_1[:, i]
            cols_0 = rows_0[:, i]

            prob_1 = (np.sum(cols_1) + self.smoothing) / (len(rows_1) + 2*self.smoothing)
            prob_0 = (np.sum(cols_0) + self.smoothing) / (len(rows_0) + 2*self.smoothing)

            probs_1.append(prob_1)
            probs_0.append(prob_0)

        self.probs_1 = np.array(probs_1)
        self.probs_0 = np.array(probs_0)

        self.p1 = self.count_1 / self.length_y
        self.p0 = self.count_0 / self.length_y

    def predict(self, X):
        preds = []

        for row in X:

            # start with priors (log form)
            score_1 = np.log(self.p1)
            score_0 = np.log(self.p0)

            for j in range(len(row)):

                if row[j] == 1:
                    score_1 += np.log(self.probs_1[j])
                    score_0 += np.log(self.probs_0[j])
                else:
                    score_1 += np.log(1 - self.probs_1[j])
                    score_0 += np.log(1 - self.probs_0[j])

            if score_1 > score_0:
                preds.append(1)
            else:
                preds.append(0)

        return np.array(preds)