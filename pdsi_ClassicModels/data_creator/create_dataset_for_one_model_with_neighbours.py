import pandas as pd
import numpy as np


# Class to create dataset for one point
class OnePointDatasetCreator:
    def __init__(
        self,
        raw_data_for_one_point,
        history_len,
        num_of_future_indexes,
        pdsi_threshold=None,
    ):
        # Raw data for concrete point
        self.raw_data_for_one_point = raw_data_for_one_point

        # History length
        self.history_len = history_len

        # Number of future indexes to predict
        # If we want to predict only closest future value
        # then we should set num_of_future_indexes = 1.
        self.num_of_future_indexes = num_of_future_indexes

        # If we solve binary classification task (drought or not), then we need threshold for classification.
        # If pdsi < threshold --> drought
        self.pdsi_threshold = pdsi_threshold

    # Target: create dataset, which consists from pairs (data, target) from raw data.
    #         Each data has fixed length.
    # Each pair is an array of type [t-history_len, ..., t-1, t, ..., t+num_of_future_indexes - 1]
    # Input: None
    # Output: created dataset
    def create_dataset_from_series(self):
        df = pd.DataFrame(self.raw_data_for_one_point)
        cols = list()

        # input sequence [t-history_len, ..., t-1]
        for i in range(self.history_len, 0, -1):
            cols.append(df.shift(i))

        # output sequence [t, ..., t+num_of_future_indexes - 1]
        for i in range(0, self.num_of_future_indexes):
            cols.append(df.shift(-i))

        # concatenating both sequences
        agg = pd.concat(cols, axis=1)

        # remove NaNs
        agg.dropna(inplace=True)

        # data for current point
        data_for_one_point = agg.values

        # if classification task transform regression targets to classification labels
        if self.pdsi_threshold != None:
            labels = (
                data_for_one_point[:, -self.num_of_future_indexes :]
                < self.pdsi_threshold
            ).astype(int)
            data_for_one_point[:, -self.num_of_future_indexes :] = labels

        return data_for_one_point


# Class to create dataset for one point with neighbours
class OnePointWithNeghboursDatasetCreator:
    def __init__(
        self,
        raw_data_for_region,
        x_cur,
        y_cur,
        filter_size,
        history_len,
        num_of_future_indexes,
        pdsi_threshold=None,
    ):
        # Raw data for all points in processing region, i.e:
        # raw_data[:, x_min:x_max, y_min:y_max]
        self.raw_data_for_region = raw_data_for_region

        # Current point to process
        self.x_cur = x_cur
        self.y_cur = y_cur

        # This is how we define neighbourhood
        # E.g. 3x3, 5x5, etc.
        self.filter_size = filter_size

        # History length
        self.history_len = history_len

        # Number of future indexes to predict
        # If we want to predict only closest future value
        # then we should set num_of_future_indexes = 1.
        self.num_of_future_indexes = num_of_future_indexes

        # If we solve binary classification task (drought or not), then we need threshold for classification.
        # If pdsi < threshold --> drought
        self.pdsi_threshold = pdsi_threshold

    # Target: find neighbours of current point with (x_cur, y_cur) coordinates
    #         using defined heighbourhood
    # Input: None
    # Output: list of neighbours [(x_1, y_1), (x_2, y_2), ...]
    def get_neighbours(self):
        # Get region properties
        _, region_width, region_height = self.raw_data_for_region.shape

        # Get filter properties
        filter_width, filter_height = self.filter_size

        # Our region coordinates (x and y) are in
        # [0, region_width] and [0, region_height] correspondingly
        # Below are min and max values for x and y coordinates of
        # neighbours for current point
        x_min = max(0, self.x_cur - filter_width // 2)
        x_max = min(region_width - 1, self.x_cur + filter_width // 2)
        y_min = max(0, self.y_cur - filter_height // 2)
        y_max = min(region_height - 1, self.y_cur + filter_height // 2)

        neighbours = []

        # Iterating over allowed values of x and y coordinates
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # Since we search for neighbours,
                # we exclude current point
                if x == self.x_cur and y == self.y_cur:
                    pass
                else:
                    neighbours.append((x, y))

        return neighbours

    # Target: create dataset, which consists from pairs (data, target) from raw data.
    #         Each data has info from current point and its neighbours.
    # Input: None
    # Output: created dataset with accounting neighbours info
    def create_dataset_with_neighbours_info(self):
        neighbours = self.get_neighbours()

        # Get data for current point
        obj = OnePointDatasetCreator(
            self.raw_data_for_region[:, self.x_cur, self.y_cur],
            self.history_len,
            self.num_of_future_indexes,
            self.pdsi_threshold,
        )

        # Separate data and labels for current point
        data_with_labels_cur_point = obj.create_dataset_from_series()
        data_for_cur_point = data_with_labels_cur_point[
            :, : -self.num_of_future_indexes
        ]
        labels = data_with_labels_cur_point[:, -self.num_of_future_indexes :]

        # Iterating over neighbours
        for neighbour in neighbours:
            # Get current neighbour coordinates
            x, y = neighbour

            # Get data for current neighbour
            obj = OnePointDatasetCreator(
                self.raw_data_for_region[:, x, y],
                self.history_len,
                self.num_of_future_indexes,
                self.pdsi_threshold,
            )
            data_for_neighbour = obj.create_dataset_from_series()

            # Concatenating histories for current point and neighbour
            data_for_cur_point = np.concatenate(
                (
                    data_for_cur_point,
                    data_for_neighbour[:, : -self.num_of_future_indexes],
                ),
                axis=1,
            )

        # Get filter properties
        filter_width, filter_height = self.filter_size

        # If number of neighbours for current point are less than maximum number of neighbours
        # than we pad remain neighboures by zeroes
        if len(neighbours) < filter_width * filter_height - 1:
            # Here is the quantity for
            zeroes_num = (
                filter_width * filter_height - len(neighbours) - 1
            ) * self.history_len
            zeroes_mas = np.zeros((data_for_cur_point.shape[0], zeroes_num))
            data_for_cur_point = np.concatenate(
                (data_for_cur_point, zeroes_mas), axis=1
            )

        # Merge current data with labels for current point
        # print("Shape of the data for current point: {}".format(data_for_cur_point.shape))
        # print("Shape of the labels for current point: {}".format(labels.shape))

        # labels = np.reshape(labels, (-1,1))
        # print("Shape of the labels AFTER for current point: {}".format(labels.shape))

        data_for_cur_point = np.concatenate((data_for_cur_point, labels), axis=1)

        # Check that shape is fit to true conditions
        # Second term 1 takes into account true label
        assert_msg = "Incorrect shape of dataset"
        cur_point_shape = data_for_cur_point.shape[1]
        true_cur_point_shape = (
            self.history_len * filter_width * filter_height + self.num_of_future_indexes
        )
        assert cur_point_shape == true_cur_point_shape, assert_msg

        return data_for_cur_point


# Class to create dataset for all points
class AllPointsDatasetCreator:
    def __init__(
        self,
        raw_data_for_all_points,
        history_len,
        num_of_future_indexes,
        time_border,
        x_min,
        x_max,
        y_min,
        y_max,
        filter_size,
        pdsi_threshold=None,
    ):
        # Iterating model over [x_min, x_max)x[y_min, y_max) points
        assert_x_msg = "X coordinate should be in range [0, {})".format(
            raw_data_for_all_points.shape[1]
        )
        assert_y_msg = "Y coordinate should be in range [0, {})".format(
            raw_data_for_all_points.shape[2]
        )
        assert (
            0 <= x_min < raw_data_for_all_points.shape[1]
            and 0 < x_max <= raw_data_for_all_points.shape[1]
        ), assert_x_msg
        assert (
            0 <= y_min < raw_data_for_all_points.shape[2]
            and 0 < y_max <= raw_data_for_all_points.shape[2]
        ), assert_y_msg

        # Region of interest
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Raw data for all points
        self.raw_data_for_all_points = raw_data_for_all_points[
            :, x_min:x_max, y_min:y_max
        ]

        # History length
        self.history_len = history_len

        # Number of future indexes to predict
        # If we want to predict only closest future value
        # then we should set num_of_future_indexes = 1.
        self.num_of_future_indexes = num_of_future_indexes

        # Time threshold, which we use to divide to train and test
        self.time_border = time_border

        # Store final arrays to train and test
        self.train_array = None
        self.test_array = None

        # Store train and test array for each point in list
        # Len of this list should be equal to (x_max - x_min)*(y_max - y_min)
        self.train_array_by_points = []
        self.test_array_by_points = []

        # If we solve binary classification task (drought or not), then we need threshold for classification.
        # If pdsi < threshold --> drought
        self.pdsi_threshold = pdsi_threshold

        # This is how we define neighbourhood
        # E.g. 3x3, 5x5, etc.
        self.filter_size = filter_size

    # Target: create train and test datasets for all points
    # Input data: None
    # Output data: None
    def create_train_and_test_datasets(self):
        # Array, which consists from time moments "t" for each pair (data, target) and for each point
        # Each element of array "times" corresponds to "current" moment of corresponding pair (data, target)
        times = [
            ind
            for ind in range(
                self.history_len - 1,
                self.raw_data_for_all_points.shape[0] - self.num_of_future_indexes,
            )
        ]

        # Find index of border pair (data, target), for which "current" moment is equal to time_border.
        # All pairs, whose "current" moments are less, than time_border,
        # will belong to train dataset. Others - to test.
        idx = times.index(self.time_border)

        x_min = 0
        x_max = self.raw_data_for_all_points.shape[1]
        y_min = 0
        y_max = self.raw_data_for_all_points.shape[2]

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Get raw data for concrete point
                raw_data_for_one_point = self.raw_data_for_all_points[:, x, y]

                # Transform raw data to supervised manner
                obj = OnePointWithNeghboursDatasetCreator(
                    self.raw_data_for_all_points,
                    x,
                    y,
                    self.filter_size,
                    self.history_len,
                    self.num_of_future_indexes,
                    self.pdsi_threshold,
                )

                data_for_one_point = obj.create_dataset_with_neighbours_info()

                # If we process the first point, than there is nothing to
                # concatenate with.
                if x == x_min and y == y_min:
                    self.train_array = data_for_one_point[:idx]
                    self.test_array = data_for_one_point[idx:]

                # Otherwise, we concatenate training and test datasets for current point
                # with training and test datasets for previous points.
                else:
                    self.train_array = np.concatenate(
                        (self.train_array, data_for_one_point[:idx]), axis=0
                    )
                    self.test_array = np.concatenate(
                        (self.test_array, data_for_one_point[idx:]), axis=0
                    )

                # Store train and test data for each point
                self.train_array_by_points.append(data_for_one_point[:idx])
                self.test_array_by_points.append(data_for_one_point[idx:])

    def get_train_array(self):
        return self.train_array

    def get_test_array(self):
        return self.test_array

    def get_train_array_by_points(self):
        assert_msg = "Length of list should be equal overall number of points!"
        assert len(self.train_array_by_points) == (self.x_max - self.x_min) * (
            self.y_max - self.y_min
        ), assert_msg

        return self.train_array_by_points

    def get_test_array_by_points(self):
        assert_msg = "Length of list should be equal overall number of points!"
        assert len(self.test_array_by_points) == (self.x_max - self.x_min) * (
            self.y_max - self.y_min
        ), assert_msg

        return self.test_array_by_points
