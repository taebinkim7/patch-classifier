from joblib import dump, load
from dwd import DWD


class DWDClassifier(DWD):
    def __init__(self, C='auto'):
        super().__init__(C)

    def train(self, X, y):
        DWD.fit(self, X, y)

    def accuracy(self, X, y):
        return DWD.score(self, X, y)

    def save(self, fpath, compress=3):
        """
        Saves to disk, see documentation for joblib.dump
        """
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

# TODO: Add svm, neural network
