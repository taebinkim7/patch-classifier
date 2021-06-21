import os
import numpy as np
import pandas as pd

from tma_classifier.Paths import Paths
from tma_classifier.classifier.models import DWDClassifier
from tma_classifier.classifier.utils import save_dataset


# TODO: Add argparse for classifier type

def train_classifier(image_type, classifier_type, save_classifier=True, save_datasets=True):
    core_centroids = pd.read_csv(os.path.join(Paths().patches_dir, 'core_centroids_' + image_type + '.csv'), index_col=0)
    core_labels = pd.read_csv(os.path.join(Paths().data_dir, 'core_labels_' + image_type + '.csv'), index_col=0)

    feats = core_centroids.to_numpy()
    label = core_labels[image_type + '_label'].to_numpy().astype(int)

    n = len(label)
    perm_idx = np.random.RandomState(seed=123).permutation(np.arange(n))
    train_idx, test_idx = perm_idx[:int(0.8 * n)], perm_idx[int(0.8 * n):]

    train_feats, test_feats = feats[train_idx], feats[test_idx]
    train_label, test_label = label[train_idx], label[test_idx]

    if classifier_type == 'dwd':
        classifier = DWDClassifier().fit(train_feats, train_label)

    acc = classifier.score(test_feats, test_label)
    print('The prediction accuracy of the trained {} on the test data is {}.'\
            .format(classifier_type.upper(), acc))

    if save_classifier:
        classifier.save(os.path.join(Paths().results_dir,
                        classifier_type + '_' + image_type))

    if save_datasets:
        dataset = {'train': [train_feats, train_label],
                   'test': [test_feats, test_label]}
        fpath = os.path.join(Paths().data_dir, 'datasets_' + image_type)
        save_dataset(dataset, fpath)

train_classifier('er', 'dwd', True, True)
