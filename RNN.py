import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float],
                Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:

    hidden_state = initial_hidden_state.copy()
    for x_t in input_sequence:
        next_hidden_state = []

        for j in range(len(hidden_state)):

            sum_input = sum(Wx[j][k] * x_t[k] for k in range(len(x_t)))
            sum_hidden = sum(Wh[j][k] * hidden_state[k] for k in range(len(hidden_state)))
            h_new = np.tanh(sum_input + sum_hidden + b[j])
            next_hidden_state.append(h_new)

        hidden_state = next_hidden_state

    return [round(h, 4) for h in hidden_state]

