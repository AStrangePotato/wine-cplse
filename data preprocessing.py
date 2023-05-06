import numpy as np
import random

dataset = np.loadtxt("winequality-white.csv", delimiter=";")

data = dataset[:, 0:11]
labels = dataset[:, 11]

td = np.empty((0, data.shape[1]), dtype=data.dtype)
tl = np.array([])
for one in range(898):
    i = random.randint(0, len(labels) - 1)
    td = np.vstack((td, data[i]))
    tl = np.append(tl, labels[i])
    data = np.delete(data, i, axis=0)
    labels = np.delete(labels, i)

    
#np.save('training_data.npy', data)
#np.save('training_labels.npy', labels)
#np.save('test_data.npy', td)
#np.save('test_labels.npy', tl)
