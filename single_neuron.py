import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float],
    bias: float) -> (list[float], float):

    probabilities = []
    for i in range(len(features)):
        z = 0
        for j in range(len(weights)):
            z += features[i][j] * weights[j]
        z += bias
        probs =  1 / (1 + math.exp(-z))
        probabilities.append(probs)


    mse = 0
    for i in range(len(labels)):
        mse += (probabilities[i] - labels[i])**2
    mse = mse/len(probabilities)

    return probabilities, mse


