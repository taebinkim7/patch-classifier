import os
import numpy as np
import pandas as pd

from tma_classifier.Paths import Paths
from tma_classifier.classifier.models import DWDClassifier
from tma_classifier.classifier.utils import save_clf_dataset


# TODO: Add argparse for classifier type (e.g., 'dwd') and level (e.g., 'core', 'subj')

def train_classifier(image_type, clf_level, clf_type, save_classifier=True, save_dataset=True):
    feats_file = os.path.join(Paths().features_dir,
                              clf_level + '_features_' + image_type + '.csv')
    labels_file = os.path.join(Paths().classification_dir,
                               clf_level + '_labels_' + image_type + '.csv')
    feats = pd.read_csv(feats_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0)

    feats = feats.to_numpy()
    labels = labels[image_type + '_labels'].to_numpy().astype(int)

    n = len(labels)
    perm_idx = np.random.RandomState(seed=111).permutation(np.arange(n))
    train_idx, test_idx = perm_idx[:int(.85 * n)], perm_idx[int(.85 * n):]

    train_feats, test_feats = feats[train_idx], feats[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    if clf_type == 'dwd':
        classifier = DWDClassifier().fit(train_feats, train_labels)

    acc_train = classifier.score(train_feats, train_labels)
    acc_test = classifier.score(test_feats, test_labels)
    print('The prediction accuracy of the trained {} on the train data is {}.'\
            .format(clf_type.upper(), acc_train))
    print('The prediction accuracy of the trained {} on the test data is {}.'\
            .format(clf_type.upper(), acc_test))

    if save_classifier:
        classifier.save(os.path.join(Paths().classification_dir,
                        clf_type + '_' + clf_level + '_' + image_type))

    if save_dataset:
        dataset = {'train': [train_feats, train_labels],
                   'test': [test_feats, test_labels]}
        fpath = os.path.join(Paths().classification_dir,
                             'clf_dataset_' + clf_level + '_' + image_type)
        save_clf_dataset(dataset, fpath)

train_classifier('er', 'subj', 'dwd', True, True)
