import math

PI = 3.14159


def power_grid_forecast(consumption_data):
    # 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
    # 2) Perform linear regression on the detrended data.
    # 3) Predict day 15's base consumption.
    # 4) Add the day 15 fluctuation back.
    # 5) Round, then add a 5% safety margin (rounded up).
    # 6) Return the final integer.

    fluctuations = []
    for i in range(1, 11):
        fluctuations_i = 10 * math.sin((2 * PI * i) / 10)
        fluctuations.append(fluctuations_i)

    without_fluctuations = []
    for i in range(len(consumption_data)):
        base_i = consumption_data[i] - fluctuations[i]
        without_fluctuations.append(base_i)

    x = list(range(1, 11))
    no_of_dataPoints = len(x)

    sum_x = sum(x)
    sum_y = sum(without_fluctuations)
    sum_xy = sum([x[i] * without_fluctuations[i] for i in range(len(x))])
    sum_x2 = sum(x[i] ** 2 for i in range(len(x)))

    m = ((no_of_dataPoints * sum_xy) - (sum_x * sum_y)) / ((no_of_dataPoints * sum_x2) - (sum_x) ** 2)

    b = (sum(without_fluctuations) - m * sum_x) / no_of_dataPoints

    base_15 = m * 15 + b
    f_15 = 10 * math.sin(2 * PI * 15) / 10
    pred_15 = base_15 + f_15

    final = math.ceil(pred_15 * 1.05)
    return final
