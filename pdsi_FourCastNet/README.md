# FourCastNet for droughts forecasting

PyTorch implementation of classifier built on NVIDIA FourCastNet model for droughts forecasting. Classification is based on [PDSI index](https://en.wikipedia.org/wiki/Palmer_drought_index), and its corresponding bins. 

<img src="https://raw.githubusercontent.com/VGrabar/Weather-Prediction-NN/multiclass/docs/pdsi_bins.png" width="400" height="250">


## Docker container launch

First, please build an image

```
docker build . -t=<docker_image_name>
```
Then please launch a container with necessary system configuration

```
docker run --mount type=bind,source=/local_path/long-term-drought-prediction/,destination=/long-term-drought-prediction/ -p <port_in>:<port_out> --memory=64g --cpuset-cpus="0-7" --gpus '"device=0"'  -it --rm --name=<docker_container_name>  <docker_image_name>
```

## Preprocessing ##
```
cd ../long-term-drought-prediction/pdsi_FourCastNet/
```

Input data, geospatial (monthly) datasets, should be downloaded as .tif-files from public sources (e.g., from Google Earth Engine) and  moved to `../data/raw` folder. For convention, file should be renamed as `region_feature.tif` (e.g., `missouri_pdsi.tif`). Then please run a script

```
python3 to_hdf5.py --region region_name --feature feature_name --endyear last_year_of_data --traintestsplit train_test_split_ratio
```
e.g.
```
python3 to_hdf5.py --region missouri --feature pdsi --endyear 2022 --traintestsplit 80
```

Note that for our five datasets from paper last year of observations is 2022, and it is set up as a default "endyear" in our parser. Preprocessed data (yearly .h5 files) will be placed in `../data/h5/train/` and `../data/h5/test/` folders. After preprocessing, please proceed to `examples/fourcastnet/` folder
```
cd examples/fourcastnet/
```

## Train ##

For training - at first, please edit configuration file `conf/config_FCN.yaml` (all essential lines - data path, forecast length, number of epochs, etc. - are commented) - and then run
```
python3 fcn_era5.py
```


## Inference ##

To run a model on test data, count metrics and save predictions - at first, please edit file `conf/config_FCN.yaml` (test data and checkpoint paths),and launch a script
```
python3 inferencer.py

