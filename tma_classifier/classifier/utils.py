import os
from joblib import dump

def save_dataset(dataset, fpath, compress=3):
    dump(dataset, fpath, compress=compress)
    
