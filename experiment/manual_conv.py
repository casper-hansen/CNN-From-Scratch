import numpy as np

x = np.array([[1, 2, 3, 4],
              [2, 1, 6, 6],
              [1, 5, 4, 3],
              [5, 4, 8, 2]])

f = np.array([[2, 3, 1],
              [1, 1, 1],
              [3, 1, 1]])

x = np.pad(x, [(1,1), (1,1)], mode='constant', constant_values=0)

slices = [x[0:3, 0:3], x[0:3, 3:6], x[3:6, 0:3], x[3:6, 3:6]]

for position in slices:
    output = np.vdot(position, f)
    print(output)