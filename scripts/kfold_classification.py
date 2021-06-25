import os
import numpy as np
import pandas as pd

from tma_classifier.Paths import Paths
from tma_classifier.classifier.models import DWDClassifier
from tma_classifier.classifier.utils import save_clf_dataset


# TODO: Add argparse for classifier type (e.g., 'dwd') and level (e.g., 'core', 'subj')
# TODO: implement 5-fold/10-fold cross validation

def train_classifier(image_type, clf_level, clf_type, n_folds, save_dataset=True):
    feats_file = os.path.join(Paths().features_dir,
                              clf_level + '_features_' + image_type + '.csv')
    labels_file = os.path.join(Paths().classification_dir,
                               clf_level + '_labels_' + image_type + '.csv')
    feats = pd.read_csv(feats_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0)

    feats = feats.to_numpy()
    labels = labels[image_type + '_label'].to_numpy().astype(int)

    if clf_type == 'dwd':
        Classifier = DWDClassifier

    n = len(labels)
    res = n % n_folds
    n_ = n - res
    perm_idx = np.random.RandomState(seed=111).permutation(np.arange(n))
    perm_idx_ = perm_idx[:-res]
    fold_idx_list = np.split(np.arange(n_), n_folds)
    acc_list = []
    for fold_idx in fold_idx_list:
        train_idx, test_idx = np.delete(perm_idx, fold_idx), perm_idx[fold_idx]
        train_feats, test_feats = feats[train_idx], feats[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        clf = Classifier().fit(train_feats, train_labels)
        acc_list.append(clf.score(test_feats, test_labels))

    acc_list = np.array(acc_list)
    kfold_acc = np.mean(acc_list)

    print('{} folds accuracy of {} is {}.'\
            .format(n_folds, clf_type.upper(), kfold_acc))

    if save_dataset:
        dataset = {'train': [train_feats, train_labels],
                   'test': [test_feats, test_labels]}
        fpath = os.path.join(Paths().classification_dir,
                             'clf_dataset_' + clf_level + '_' + image_type)
        save_clf_dataset(dataset, fpath)

train_classifier('er', 'subj', 'dwd', 5, True)
