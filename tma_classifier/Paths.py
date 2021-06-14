import os


class Paths(object):
    """
    Contains paths to directories used in the analysis.
    The user should modify data_dir; everything else should work from there.
    """
    def __init__(self):

        # top level data directory for the analysis
        # The user should modify this attribute before installing the package!
        self.data_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9830'

        self.patches_dir = os.path.join(self.data_dir, 'patches')
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.results_dir = os.path.join(self.data_dir, 'results')

    def make_directories(self):
        """
        Creates the top level data directories.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.patches_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
