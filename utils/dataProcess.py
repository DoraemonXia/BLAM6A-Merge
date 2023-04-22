import random
import numpy as np
def shuffleData(X, y):
    X = np.array(X)
    y = np.array(y)
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    X = X.tolist()
    y = y.tolist()
    return X, y;