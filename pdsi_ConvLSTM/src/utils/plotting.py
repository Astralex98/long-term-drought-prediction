import os
import random

import numpy as np
import seaborn as sns
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def make_heatmap(table, filename="rocauc_spatial.png", size=(8, 6)):
    fig = Figure(figsize=size, frameon=True)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax = sns.heatmap(table, vmin=0.0, vmax=1.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    full_path = os.path.expanduser(filename)
    ax.figure.savefig(full_path)
    ax.cla()

    return full_path

def make_pred_vs_target_plot(
    preds,
    targets,
    title="Comparison",
    size=(8, 6),
    xlabel=None,
    xlabel_rotate=45,
    ylabel=None,
    ylabel_rotate=0,
    filename="forecasts.png",
):
    fig = Figure(figsize=size, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    x_length = targets.shape[1]
    y_length = targets.shape[2]
    x_random = random.choice(list(range(x_length)))
    y_random = random.choice(list(range(y_length)))
    targets = targets.cpu()
    targets = torch.mean(targets, dim=[1, 2])
    preds = preds.cpu()
    preds = torch.mean(preds, dim=[1, 2])
    time_periods = np.arange(0, targets.shape[0])
    ax.plot(time_periods, targets, "g-", label="actual")
    ax.plot(time_periods, preds, "b--", label="predictions")
    ax.legend()

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    filename = os.path.expanduser(filename)
    fig.savefig(filename)
    return fig
