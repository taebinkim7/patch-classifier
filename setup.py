from setuptools import setup, find_packages

# the analysis was done with Python version 3.7.2.
# glob, os, shutil, datetime, codecs, copy, numbers

install_requires = ['numpy==1.18.1',
                    'matplotlib==3.1.3',
                    'pandas==1.0.1',
                    'tqdm==4.42.1',
                    'torch==1.5.0',
                    'torchvision==0.6.0',
                    'scikit-image==0.16.2',
                    'joblib==0.14.1'
                    ]

setup(name='tma_classifier',
      version='0.0.1',
      description='Code to reproduce breast cancer TMA classification',
      author='Taebin Kim',
      author_email='taebinkim@unc.edu',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=install_requires,
      zip_safe=False)
