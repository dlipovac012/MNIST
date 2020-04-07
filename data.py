import numpy as np
from sklearn.datasets import fetch_openml

print("Fetching data...")
mnist = fetch_openml('mnist_784', version=1, cache=True)
print("Data fetching completed!")

X, y = mnist['data'], mnist['target']

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)