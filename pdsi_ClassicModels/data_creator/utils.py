import pandas as pd
import torch
import tqdm
import pathlib


def create_celled_data(
    dataset_name: str,
    time_col: str = "date",
    event_col: str = "value",
    x_col: str = "x",
    y_col: str = "y",
):
    """
    Transform geospatial dataset from .csv to torch_tensor (num_of_months, H, W)
    """
    data_path = pathlib.Path(
        "../data/preprocessed/",
        dataset_name,
    )
    celled_data_path = pathlib.Path(
        "../data/celled/",
        dataset_name,
    )
    if celled_data_path.is_file():
        print(f"Torch tensor {celled_data_path} already exists")
        celled_data = torch.load(celled_data_path)
        return celled_data

    df = pd.read_csv(data_path)
    # time from str to int
    df[time_col] = df[time_col].apply(
        lambda x: int(x.split("-")[0]) * 12 + int(x.split("-")[1])
    )
    df.sort_values(by=[time_col], ascending=True, inplace=True)
    df = df[[event_col, x_col, y_col, time_col]]

    indicies = range(df.shape[0])
    start_date = df[time_col].min()  # int(df[time_col][indicies[0]])
    finish_date = df[time_col].max()  # int(df[time_col][indicies[-1]])
    n_cells_hor = df[x_col].max() - df[x_col].min() + 1
    n_cells_ver = df[y_col].max() - df[y_col].min() + 1
    celled_data = torch.zeros([finish_date - start_date + 1, n_cells_hor, n_cells_ver])

    for i in tqdm.tqdm(indicies):
        x = int(df[x_col][i])
        y = int(df[y_col][i])
        celled_data[int(df[time_col][i]) - start_date, x, y] = df[event_col][i]

    torch.save(celled_data, celled_data_path)

    return celled_data
