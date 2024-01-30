from typing import List

import numpy as np
import pandas as pd
import pathlib
import torch

import tqdm


def create_celled_data(
    data_path,
    dataset_name,
    time_col: str = "date",
    event_col: str = "value",
    x_col: str = "x",
    y_col: str = "y",
):
    data_path = pathlib.Path(
        data_path,
        dataset_name,
    )

    df = pd.read_csv(data_path)
    # time from str to int
    df[time_col] = df[time_col].apply(lambda x: int(x.split("-")[0]) * 12 + int(x.split("-")[1]))
    df.sort_values(by=[time_col], ascending=True, inplace=True)
    df = df[[y_col, x_col, event_col, time_col]]

    indicies = range(df.shape[0])
    n_cells_hor = df[x_col].max() - df[x_col].min() + 1
    n_cells_ver = df[y_col].max() - df[y_col].min() + 1
    start_date = df[time_col].min()
    end_date = df[time_col].max()
    print(start_date)
    print(end_date)
    celled_data = torch.zeros([end_date - start_date + 1, n_cells_hor, n_cells_ver])

    for i in tqdm.tqdm(indicies):
        x = int(df[x_col][i])
        y = int(df[y_col][i])
        celled_data[int(df[time_col][i]) - start_date, x, y] = df[event_col][i]

    return celled_data
