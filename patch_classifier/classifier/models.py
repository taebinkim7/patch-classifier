import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report

from joblib import dump, load
from dwd import DWD

from patch_classifier.classifier.nn_models import MLP


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

# include in script
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = MLP()
# model.to(device)

class MLPClassifier():


# TODO: Add SVM, MLP
