import torch
import os
import numpy as np
import pandas as pd

from torchvision.transforms import Normalize, ToTensor, Compose

from patch_classifier.patches.PatchGrid import PatchGrid
from patch_classifier.patches.patch_features import compute_patch_features
from patch_classifier.patches.cnn_models import load_cnn_model


# CNN feature extraction model
model = load_cnn_model()

#######################
# get patches dataset #
#######################

# compute the backgorund mask for each image, break into patches, throw out
# patches which have too much background

def patch_feat_extraction(paths, image_type, patch_size=200,
                          max_prop_background=.9):

    os.makedirs(paths.features_dir, exist_ok=True)
    patch_kws = {'paths': paths,
                 'patch_size': patch_size,
                 'pad_image': 'div_' + str(patch_size),
                 'max_prop_background': .9,
                 'threshold_algo': 'triangle_otsu',
                 'image_type': image_type}

    patch_dataset = PatchGrid(**patch_kws)
    patch_dataset.make_patch_grid()
    patch_dataset.compute_pixel_stats(image_limit=10)
    patch_dataset.save(os.path.join(paths.features_dir,
                                    'patch_dataset_' + image_type))

    ##############################
    # Extract patch CNN features #
    ##############################

    # patch image processing
    # center and scale channels
    channel_avg = patch_dataset.pixel_stats_['avg'] / 255
    channel_std = np.sqrt(patch_dataset.pixel_stats_['var']) / 255
    patch_transformer = Compose([ToTensor(),
                                 Normalize(mean=channel_avg, std=channel_std)])

    fpath = os.path.join(paths.features_dir,
                         'patch_features_' + image_type + '.csv')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    compute_patch_features(image_type=image_type,
                           patch_dataset=patch_dataset,
                           model=model,
                           fpath=fpath,
                           patch_transformer=patch_transformer,
                           device=device)

    # make copy of patch features to generate core features and subject features
    patch_feats = pd.read_csv(fpath, index_col=['image', 'patch_idx'])
    patch_feats_ = patch_feats.copy()

    ######################
    # save core features #
    ######################

    # core ID:  <subject ID>_core<core no.>
    core_ids = list(map(lambda x: x[0].split('_')[0] + '_' + x[0].split('_')[1],
                        patch_feats_.index))
    patch_feats_['core'] = core_ids
    core_feats = patch_feats_.groupby('core').mean()
    core_feats.to_csv(os.path.join(paths.features_dir,
                                   'core_features_' + image_type + '.csv'))

    #########################
    # save subject features #
    #########################

    subj_ids = list(map(lambda x: x[0].split('_')[0], patch_feats_.index))
    patch_feats_['subject'] = subj_ids
    subj_feats = patch_feats_.groupby('subject').mean()
    subj_feats.to_csv(os.path.join(paths.features_dir,
                                   'subj_features_' + image_type + '.csv'))
