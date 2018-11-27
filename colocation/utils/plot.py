"""Utility Function for plotting
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_each_series(datas, title, indicator_lines=False):
    fig = plt.figure(figsize=(6, 2 * len(datas)))

    for i, (label, data) in enumerate(datas.items()):
        plt.subplot(len(datas), 1, i + 1)
        plt.plot(data)
        if indicator_lines:
            mean = np.mean(data)
            std = np.std(data)
            plt.hlines(
                [mean, mean - std, mean + std, mean - 2 * std, mean + 2 * std],
                xmin=0,
                xmax=len(data),
                colors=["red", "yellow", "yellow", "green", "green"],
                zorder=5,
            )
        plt.title(label)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
