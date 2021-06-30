import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

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

class NNClassifier(n.Mo)

# TODO: Add SVM, neural network
