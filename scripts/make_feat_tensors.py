import os
import numpy as np
import pandas as pd

from joblib import dump
from patch_classifier.Paths import Paths

def make_feat_tensors(image_type, grid_size=6):
    feats_path = os.path.join(Paths().features_dir,
                              'patch_features_' + image_type + '.csv')
    patch_feats = pd.read_csv(feats_path, index_col=['image', 'patch_idx'])
    tensors = {}
    core_ids_ = np.unique(patch_feats.index.get_level_values('image'))
    patch_feats = patch_feats.to_numpy()
    patch_set_list = np.split(patch_feats, len(core_ids_))
    for id_, patch_set in zip(core_ids_, patch_set_list):
        id = id_.split('_')[0] + '_' + id_.split('_')[1]
        tensor = np.array(np.split(patch_set, grid_size))
        tensors[id] = tensor

    tensors_path = os.path.join(Paths().features_dir,
                                'core_tensors_' + image_type)
    dump(tensors, tensors_path)

make_feat_tensors('er')
