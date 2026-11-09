"""SVC initial configuration: keep baseline; edit candidate."""
from sklearn.svm import SVC

def build_baseline():
    return SVC(), True

def build_candidate():
    # Edit C/gamma/kernel, or return scaling=False for experiments.
    return SVC(), True
