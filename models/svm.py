"""Provides a SVM model using sklearn."""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def use(X_train, y_train, X_test, y_test):
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
    clf = SVC(gamma='auto', kernel='rbf')
    # Train model on X_train providing labels
    print('Model training..')
    clf.fit(X=X_train, y=y_train)
    print('Model trained..')
    # Test model
    print('Model testing..')
    score = clf.score(X=X_test, y=y_test)
    print(score)
