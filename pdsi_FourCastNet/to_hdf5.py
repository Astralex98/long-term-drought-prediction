from osgeo import gdal
import numpy as np
import tqdm
import argparse
import h5py
import os


parser = argparse.ArgumentParser(description="preprocess tif files to h5")
parser.add_argument("--region", type=str, default="missouri", help="name of region")
parser.add_argument("--feature", type=str, default="pdsi", help="name of feature")
parser.add_argument("--endyear", type=int, default=2022, help="end year of data")
parser.add_argument("--traintestsplit", type=int, default=80, help="percent of train data")


args = parser.parse_args()
region = args.region
feature = args.feature
end_year = args.endyear
train_test_split = args.traintestsplit / 100
path = "/long-term-drought-prediction/data/h5/" + feature + "_" + region + "_h5"
try: 
    os.mkdir(path) 
    os.mkdir(path + "/train/")
    os.mkdir(path + "/test/")
    os.mkdir(path + "/stats/")
except OSError as error: 
    print(error)  


ds = gdal.Open("/long-term-drought-prediction/data/raw/" + region + "_" + feature + ".tif")
print(ds.GetMetadata())
print("number of months ", ds.RasterCount)
print("x dim ", ds.RasterXSize)
print("y dim ", ds.RasterYSize)

driver = gdal.GetDriverByName('HDF4')

start_year = end_year - ds.RasterCount // 12 + 1
first_year = start_year

num_of_months = ds.RasterCount
xsize = ds.RasterXSize
ysize = ds.RasterYSize

temp_year = np.zeros((12, 1, ysize, xsize))
all_data = np.zeros((num_of_months, 1, ysize, xsize))

for i in tqdm.tqdm(range(1, num_of_months)):
    
    if i % 12 == 1:
        temp_year = np.zeros((12, 1, ysize, xsize))

    band = ds.GetRasterBand(i)
    data = band.ReadAsArray()
    # pdsi needs to be normalized by 100
    temp_year[i % 12 - 1][0] = data/100
    all_data[i - 1][0] = data/100

    if i % 12 == 0:
        # save file
        print(f"saving year {start_year}")
        if start_year - first_year > train_test_split * (end_year - first_year):
            h5f = h5py.File(path + "/test/" + str(start_year) + '.h5', 'w')
        else:
            h5f = h5py.File(path + "/train/" + str(start_year) + '.h5', 'w')

        temp_np = 1 - np.digitize(temp_year, np.array([-2]))
        print("annual stats")
        unique_elements, counts_elements = np.unique(temp_np, return_counts=True)
        len_np = 12 * 1 * ysize * xsize
        print(np.asarray((unique_elements, counts_elements/len_np)))
        h5f.create_dataset('fields', data=temp_year)
        print(h5f.keys())
        start_year += 1


temp_np = 1 - np.digitize(all_data, np.array([-2]))
print("total stats")
unique_elements, counts_elements = np.unique(temp_np, return_counts=True)
len_np = num_of_months * 1 * ysize * xsize
print(np.asarray((unique_elements, counts_elements/len_np)))
print(f"{region} global stats")
print(f"mean: {np.mean(all_data)}")
print(f"std: {np.std(all_data)}")
num_of_channels = 1
global_means = np.zeros((1, num_of_channels, 1, 1))
global_stds = np.zeros((1, num_of_channels, 1, 1))
global_means[0, 0, 0, 0] = np.mean(all_data)
global_stds[0, 0, 0, 0] = np.std(all_data)
np.save(path + "/stats/global_means.npy", global_means)
np.save(path + "/stats/global_stds.npy", global_stds)
