import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from data import X, X_train, y_train_5
from validation import validate

model = SGDClassifier(random_state=42)

some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.show()

# cast string to integer

print("Training the model...")
model.fit(X_train, y_train_5)

is_five = model.predict([some_digit])

validate(model)

print(is_five)
