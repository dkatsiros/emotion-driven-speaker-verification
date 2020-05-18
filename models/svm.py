"""Provides a SVM model using sklearn."""

from sklearn.svm import NuSVC
import numpy as np
import random
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from utils.emodb import get_classes
from plotting.metrics import plot_confusion_matrix


def use(X_train, y_train, X_test, y_test, oversampling=False, pca=False):
    """Normalize data and then use the SVM model."""
    # Create standard scaler.
    scaler = StandardScaler()
    # Fit scaler to train data only, as we should know
    # nothing about the test distribution
    scaler.fit(X_train)
    # Transform both train and test set
    scaler.transform(X_train)
    scaler.transform(X_test)
    # Create classifier
    import random
    clf = NuSVC(kernel='rbf', gamma='scale', random_state=random.randint(0,100))
    # Before training oversample
    if oversampling is True:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    # Train model on X_train providing labels
    print('Model training..')
    clf.fit(X=X_train, y=y_train)
    print('Model trained..')
    # Test model
    print('Model testing..')
    y_pred = clf.predict(X=X_test)
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Get dataset classes
    classes = get_classes()
    # Save confusion matrix
    plot_confusion_matrix(cm=conf_matrix,
                          classes=classes,
                          filename=('svm_unbalanced_1'
                           if oversampling is False else 'svm_balanced_1'))
    # Print report
    print(classification_report(y_test, y_pred, target_names=classes))


def use_svm_cv(X, y, oversampling=False, pca=False):
    """Normalize data and then use the SVM model."""
    # Create standard scaler.
    scaler = StandardScaler()
    # Fit scaler to train data only, as we should know
    # nothing about the test distribution
    scaler.fit(X)
    # Transform both train and test set
    scaler.transform(X)
    # Create classifier
    clf = NuSVC(kernel='rbf', gamma='scale', random_state=random.randint(0,100))
    scoreSVM = cross_validate(clf, X, y, cv=7)
    print('RBF SVM scored', 100 *
          np.mean(scoreSVM['test_score']), '% in 7-fold cross-validation.')
    # transform the results of the “one-versus-one” classifiers to a “one-vs-rest”
    # clf.decision_function_shape = "ovr"
    # Before training oversample
    if oversampling is True:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
