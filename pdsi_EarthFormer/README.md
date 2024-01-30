This code is mainly based on original EarthFormer GitHub: https://github.com/amazon-science/earth-forecasting-transformer

## Getting started

- First, please build a Docker image
```
docker build -t earthformer_image .
```

- Then please launch a container with necessary system configuration 

```
docker run -it --rm --mount type=bind,source=/local_path/long-term-drought-prediction/,destination=/long-term-drought-prediction/ --memory=64g --memory-swap=64g --cpuset-cpus=0-5 --gpus '"device=0,1"' -p 8001:8001 --gpus 1 --name "earthformer_container" earthformer_image
```

## Preprocessing ##

The EarthFormer model takes **.csv** files as an input. In order to process raw **.tif** files from Google Earth Engine please proceed to `../preprocessing/` folder. 
Preprocessed data (both as **.csv** and **.npy** files) will be placed in `../data/preprocessed/` folder.

## Training and validating model

Please proceed to the folder of main script

```
cd ../long-term-drought-prediction/pdsi_EarthFormer/scripts/cuboid_transformer/pdsi/
```
and launch it with the parameters from configuration file
```
python3 train_cuboid_pdsi.py --cfg cfg.yaml --ckpt_name last.ckpt --save region_name
```

The configuration file **cfg.yaml** is in the following folder: `scripts/cuboid_transformer/pdsi/`
Most of hyperparameters in this file are from original EarthFormer code and our hyperparameters are commented. You can change region name, boundaries, input history for model and forecasting horizon.


