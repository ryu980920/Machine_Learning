"""MLP initial configuration."""
from sklearn.neural_network import MLPClassifier

def build_baseline():
    return MLPClassifier(random_state=42), True

def build_candidate():
    # Edit hidden_layer_sizes, alpha, max_iter, etc.
    return MLPClassifier(random_state=42), True
