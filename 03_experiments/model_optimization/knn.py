"""KNN initial configuration."""
from sklearn.neighbors import KNeighborsClassifier

def build_baseline():
    return KNeighborsClassifier(), True

def build_candidate():
    # Edit n_neighbors, weights, metric, etc.
    return KNeighborsClassifier(), True
