# %load T1_P1.py
import numpy as np

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])

new_data = np.asarray(data)

def k(x_1, x_2, W):
    return np.exp(-1 * np.matmul(np.matmul(np.transpose(x_1 - x_2), W), (x_1 - x_2)))

def f(x, W):
    num = 0
    denom = 0
    for row in new_data:
        if not np.array_equal(row[:2], x):
            num += k(row[:2], x, W) * row[2]
            denom += k(row[:2], x, W)
    return num / denom
     
    ##return sum(k(row[:2], x, W) * row[2] for row in new_data) / sum(k(row[:2], x, W) for row in new_data)

def compute_loss(W):
    ## TO DO

    loss = sum((row[2]-f(row[:2], W))**2 for row in new_data)
    return loss


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))