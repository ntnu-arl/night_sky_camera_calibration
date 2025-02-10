import matplotlib.pyplot as plt
import numpy as np


def plot_night_sky(coords, vmags, width, height, dpi=1000, crosshair=True):
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    _draw_night_sky(ax, coords, vmags, width, height, crosshair)
    fig.add_axes(ax)
    return fig


def save_night_sky(coords, vmags, width, height, path, crosshair=True):
    fig = plot_night_sky(coords, vmags, width, height, 1000, crosshair)
    fig.savefig(path)


def _apparent_magnitude_to_alpha(vmags, k=3):
    m = np.power(10, -0.4 * vmags)
    m = (1 + k) * m / (1 + k * m * m)
    return np.clip(m, 0, 1)


def _draw_crosshair(ax, width, height, color="white"):
    ax.plot([0, width], [height / 2, height / 2], color=color, alpha=0.15, linestyle="--", linewidth=0.5)
    ax.plot([width / 2, width / 2], [0, height], color=color, alpha=0.15, linestyle="--", linewidth=0.5)


def _draw_night_sky(ax, coords, vmags, width, height, crosshair=True):
    alpha = _apparent_magnitude_to_alpha(vmags)
    ax.set_facecolor("black")
    ax.scatter(coords[0], coords[1], s=0.5, color="white", alpha=alpha)
    if crosshair:
        _draw_crosshair(ax, width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.set_aspect("equal")
