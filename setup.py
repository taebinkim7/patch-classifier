from setuptools import setup, find_packages
# the analysis was done with Python version 3.7.2.

install_requires = ['numpy',
                    'matplotlib',
                    'pandas',
                    'tqdm',
                    'joblib',
                    'dwd',
                    'scikit-image==0.16.2',
                    'torch==1.5.0',
                    'torchvision==0.6.0',
                    'efficientnet_pytorch'
                    ]

setup(name='patch_classifier',
      version='0.0.1',
      description='Code to reproduce patch-based image classification',
      author='Taebin Kim',
      author_email='taebinkim@unc.edu',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=install_requires,
      zip_safe=False)
