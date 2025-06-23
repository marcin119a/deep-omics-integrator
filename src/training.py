from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def train_model_kfold(model_fn, X_all, y, class_weights, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    histories, scores = [], []
    for train_idx, test_idx in kf.split(X_all[0]):
        X_train = [x[train_idx] for x in X_all]
        X_test = [x[test_idx] for x in X_all]
        y_train, y_test = y[train_idx], y[test_idx]
        model = model_fn()
        history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2,
                            class_weight=class_weights, verbose=0)
        acc = model.evaluate(X_test, y_test, verbose=0)[1]
        histories.append(history.history)
        scores.append(acc)
    return histories, scores