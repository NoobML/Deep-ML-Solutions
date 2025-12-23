import numpy as np

def relu(z: float) -> float:
    if z > 0:
        return z
    else:
        return 0


print(relu(2))
print(0)
print(-1)