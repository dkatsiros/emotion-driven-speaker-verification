"""Provides a SVM model using sklearn."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN

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
    clf = KNeighborsClassifier(n_neighbors=5)

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
                          filename=('knn_balanced_SMOTE'
                          if oversampling is True else 'knn_unbalanced'))
    # Print report
    print(classification_report(y_test, y_pred, target_names=classes))
