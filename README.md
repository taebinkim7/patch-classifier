# Patch-based image Classification

This repository implements patch-based image classification.

# Instructions to run the code

### 1. Setup data directory

patch-classifier/Paths.py has instructions for setting up the data directory.

### 2. Install code

Download the github repository with
```
git clone https://github.com/taebinkim7/patch-classifier.git
```
Change the folder path in tma-classifier/Paths.py to match the data directories on your computer.

Using python >= 3.7, install the package `patch-classifier` by running
```
pip install -e .
```

### 3. Image patch feature extraction

```
python scripts/patch_feat_extraction.py
```

This step extracts CNN features from each image patch and may take a few hours. If a GPU is available it will automatically be used.

### 4. Binary classification

```
python scripts/classification.py
```
