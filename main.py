import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

X, y = mnist['data'], mnist['target']

# some_digit = X[3]
# some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.show()

# cast string to integer
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print(y_train[:10])

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
