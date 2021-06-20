from joblib import dump, load
from dwd import DWD


class Classifier(object):
    def save(self, fpath, compress=3):
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

class DWDClassifier(DWD, Classifier):
    def __init__(self, C='auto'):
        super().__init__(C)

    def train(self, X, y):
        DWD.fit(self, X, y)

    def accuracy(self, X, y):
        return DWD.score(self, X, y)
    #
    # def save(self, fpath, compress=3):
    #     dump(self, fpath, compress=compress)
    #
    # @classmethod
    # def load(cls, fpath):
    #     return load(fpath)

# TODO: Add SVM, neural network
