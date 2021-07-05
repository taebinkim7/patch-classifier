import os
import numpy as np
import pandas as pd

from joblib import dump
from tqdm import tqdm

from patch_classifier.Paths import Paths
from patch_classifier.classifier.models import DWDClassifier
from patch_classifier.classifier.utils import save_clf_dataset


# TODO: Add argparse for classifier type (e.g., 'dwd') and level (e.g., 'core', 'subj')

def classification(image_type, clf_level, clf_type, seed=None, save_classifier=False, save_dataset=False):
    feats_file = os.path.join(Paths().features_dir,
                              clf_level + '_features_' + image_type + '.csv')
    labels_file = os.path.join(Paths().classification_dir,
                               clf_level + '_labels_' + image_type + '.csv')
    feats = pd.read_csv(feats_file, index_col=0)
    labels = pd.read_csv(labels_file, ` index_col=0)

    feats = feats.to_numpy()
    labels = labels[image_type + '_label'].to_numpy().astype(int)

    n = len(labels)
    perm_idx = np.random.RandomState(seed=seed).permutation(np.arange(n))
    train_idx, test_idx = perm_idx[:int(.8 * n)], perm_idx[int(.8 * n):]

    train_feats, test_feats = feats[train_idx], feats[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    if clf_type == 'dwd':
        classifier = DWDClassifier().fit(train_feats, train_labels)

    train_acc = classifier.score(train_feats, train_labels)
    test_acc = classifier.score(test_feats, test_labels)
    print('The prediction accuracy of the trained {} on the train data is {}.'\
            .format(clf_type.upper(), train_acc))
    print('The prediction accuracy of the trained {} on the test data is {}.'\
            .format(clf_type.upper(), test_acc))

    if save_classifier:
        classifier.save(os.path.join(Paths().classification_dir,
                        clf_type + '_' + clf_level + '_' + image_type))

    if save_dataset:
        dataset = {'train': [train_feats, train_labels],
                   'test': [test_feats, test_labels]}
        fpath = os.path.join(Paths().classification_dir,
                             'clf_dataset_' + clf_level + '_' + image_type)
        save_clf_dataset(dataset, fpath)

    return train_acc, test_acc

test_acc_list = []
for i in tqdm(range(100)):
    _, test_acc = classification('er', 'subj', 'dwd')
    test_acc_list.append(test_acc)

dump(test_acc_list, os.path.join(Paths().classification_dir, 'test_acc_list'))
mean_test_acc = np.mean(test_acc_list)
print(mean_test_acc)
