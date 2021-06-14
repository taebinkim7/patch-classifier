# Breast Cancer TMA Classification

# Instructions to run the code

### 1. Setup data directories

tma-classifier/Paths.py has instructions for setting up the data directory.

### 2. Install code

Download the github repository with
```
git clone https://github.com/taebinkim7/tma-classifier.git
```
Change the folder path in tma-classifier/Paths.py to match the data directories on your computer.

Using python 3.7.2., (e.g. `conda create -n tma python=3.7.2`, `conda activate tma`) install the package `tma-classifier` by running
```
pip install -e .
```

### 3. Image patch feature extraction

```
python scripts/patch_feat_extraction.py
```

This step extracts CNN features from each image patch and may take a few hours. If a GPU is available it will automatically be used.
