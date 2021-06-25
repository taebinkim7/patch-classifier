from glob import glob
import os
from tma_classifier.Paths import Paths


def get_avail_images(image_type='er'):
    """
    Returns a list of the file names of all the images available in the
    image directory.
    Parameters
    ----------
    image_type (str): the type of the image to return. Must be one of
        ['he', 'er', 'pr', 'her2']
    """
    assert image_type in ['he', 'er', 'pr', 'her2']

    image_list = glob(os.path.join(Paths().image_dir, image_type, '*'))

    return [os.path.basename(image) for image in image_list]
