from glob import glob
import os


def get_avail_images(paths, image_type):
    """
    Returns a list of the file names of all the images available in the
    image directory.
    Parameters
    ----------
    image_type (str): the type of the image to return.
    """
    image_list = glob(os.path.join(paths.images_dir, image_type, '*'))

    return [os.path.basename(image) for image in image_list]
