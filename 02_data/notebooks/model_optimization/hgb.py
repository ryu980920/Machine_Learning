"""HistGradientBoosting initial configuration."""
from sklearn.ensemble import HistGradientBoostingClassifier

def build_baseline():
    return HistGradientBoostingClassifier(random_state=42), False

def build_candidate():
    # Edit max_iter, learning_rate, max_leaf_nodes, etc.
    return HistGradientBoostingClassifier(random_state=42), False
