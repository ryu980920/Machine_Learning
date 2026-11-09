"""Extra Trees initial configuration."""
from sklearn.ensemble import ExtraTreesClassifier

def build_baseline():
    return ExtraTreesClassifier(random_state=42), False

def build_candidate():
    # Edit n_estimators, max_depth, etc.
    return ExtraTreesClassifier(random_state=42), False
