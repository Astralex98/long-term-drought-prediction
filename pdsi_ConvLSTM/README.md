# ConvLSTM for droughts forecasting

PyTorch Lightning implementation of binary classifier built on Convolutional LSTM for droughts forecasting. Classification is based on [PDSI index](https://en.wikipedia.org/wiki/Palmer_drought_index), and its corresponding bins. 

<img src="https://raw.githubusercontent.com/VGrabar/Weather-Prediction-NN/multiclass/docs/pdsi_bins.png" width="400" height="250">

We solve drought forecasting as binary classification problem, where drought threshold (by default, -2) can be adjusted in a main config file

## Docker container launch

First, please build an image

```
docker build . -t=<docker_image_name>
```
Then please launch a container with necessary system configuration

```
docker run --mount type=bind,source=/<local_path>/long-term-drought-prediction/,destination=/long-term-drought-prediction/ -p <port_in>:<port_out> --memory=64g --cpuset-cpus="0-7" --gpus '"device=0"'  -it --rm --name=<docker_container_name>  <docker_image_name>
```

## Preprocessing ##

The ConvLSTM model takes **.csv** files as an input. In order to process raw **.tif** files from Google Earth Engine please proceed to `../preprocessing/` folder. 
Preprocessed data (both as **.csv** and **.npy** files) will be placed in `../data/preprocessed/` folder.

## Train ##
```
cd ../long-term-drought-prediction/pdsi_ConvLSTM/
```

For training - at first, please edit configs of datamodule and model (if it's necessary), edit parameters of the script (e.g., path to date in train.yaml) - and then run
```
python3 train.py --config-name=train.yaml
```
Experiments could be tracked via Comet ML - please add your token to a logger config file (`configs/logger/comet.yaml`) or export it as an enviromental variable (`export COMET_API_TOKEN="your_comet_api_token"`) and run 
```
python3 train.py --config-name=train.yaml logger=comet.yaml
```

## Inference ##

To run a model on test data, count metrics and save predictions - at first, please edit configs of datamodule and model (if it's necessary), add a path to saved checkpoint, and launch a script
```
python3 run.py --config-name=test.yaml

