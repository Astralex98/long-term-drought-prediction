# Preprocessing


## Docker container launch
First, please build an image

```
docker build . -t=<docker_image_name>
```
Then please launch a container with necessary system configuration

```
docker run --mount type=bind,source=/<local_path>/long-term-drought-prediction/,destination=/long-term-drought-prediction/ -p <port_in>:<port_out> --memory=64g --cpuset-cpus="0-7" --gpus '"device=0"'  -it --rm --name=<docker_container_name>  <docker_image_name>
```

## Main script ##
```
cd ../long-term-drought-prediction/preprocessing/
```

Input data, geospatial (monthly) datasets, should be downloaded as .tif-files from public sources (e.g., from Google Earth Engine) and  moved to `../data/raw` folder. For convention, file should be renamed as `region_feature.tif` (e.g., `missouri_pdsi.tif`). Then please run a script

```
python3 preprocess.py --region region_name --band feature_name --endyear last_year_of_data --endmonth last_month_of_data
```
e.g.
```
python3 preprocess.py --region missouri --band pdsi --endyear 2022 --endmonth 12
```

Note that for our five datasets from paper last year and last month of observations are 2022 and 12 correspondingly, and they are set up as a default "endyear" and a default "endmonth" in our parser. Preprocessed data (both as .csv and .npy files) will be placed in `../data/preprocessed` folder.
