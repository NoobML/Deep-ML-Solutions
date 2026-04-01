import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round

	# converting the X into a numpy array so its easy for transpose
	X_transpose = np.transpose(X)

	part1 = np.linalg.inv(np.dot(X_transpose, X))
	part2 = np.dot(X_transpose, y)
	theta = np.dot(part1, part2)
	return np.round(theta, 4)