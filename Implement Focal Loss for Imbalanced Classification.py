import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to avoid log(0)
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)

    losses = []

    # Loop through each sample
    for true, pred in zip(y_true, y_pred):
        # Step 1: get pt
        pt = pred[true]

        # Step 2: cross entropy
        ce = -np.log(pt)

        # Step 3: focal weight
        weight = (1 - pt) ** gamma

        # Step 4: apply alpha if provided
        if alpha is not None:
            alpha_t = alpha[true]
            loss = alpha_t * weight * ce
        else:
            loss = weight * ce

        losses.append(loss)

    # Step 5: return average loss
    return float(np.mean(losses))