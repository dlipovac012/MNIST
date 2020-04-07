from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from data import X_train, y_train_5

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

def validate(model):
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(model)
        X_train_folds = X_train[train_index]
        t_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, t_train_folds)
        y_prediction = clone_clf.predict(X_test_fold)
        n_correct = sum(y_prediction == y_test_fold)

        print(n_correct / len(y_prediction))