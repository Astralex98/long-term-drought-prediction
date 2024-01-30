## Preprocessing ##

These models take **.csv** files as an input. In order to process raw **.tif** files from Google Earth Engine please proceed to `../preprocessing/` folder. 
Preprocessed data (both as **.csv** and **.npy** files) will be placed in `../data/preprocessed/` folder.

## Train and test ##

There are two ways to use code:

- Colab versions
- Local versions

For your convenience you can just use colab versions:

1. XGBoost: https://colab.research.google.com/drive/1g3sMJG40l3qPDjcEMTA7CK_URopaQwwp?usp=sharing
2. Linear model: https://colab.research.google.com/drive/1io-0QVIkbqiL8aZDBPeCFCW8UVdgQKGi?usp=sharing

To use local version please follow next steps:

1. Create conda environment and install dependencies from requirements.txt
```
conda create -n classic python=3.8
conda activate classic
pip install -r requirements.txt
```
2. Launch jupyter notebook with created environment from step 1.

