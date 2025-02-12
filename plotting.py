import matplotlib.pyplot as plt
import numpy as np


def plot_sources(pred_sources, sources, width, height, margin=300):
    margin = max(margin, 0)

    mask = np.logical_and(
        np.logical_and(pred_sources[0] >= -margin, pred_sources[0] <= width + margin),
        np.logical_and(pred_sources[1] >= -margin, pred_sources[1] <= height + margin)
    )
    pred_sources = pred_sources[:, mask]

    fig, ax = plt.subplots()
    ax.scatter(pred_sources[0], pred_sources[1], alpha=0.5)
    ax.scatter(sources[0], sources[1], alpha=0.5)
    if margin > 0:
        _draw_rectangle(ax, 0, 0, width, height)
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(-margin, height + margin)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def plot_matches(pred_sources, sources, matches, width, height, margin=300):
    margin = max(margin, 0)

    src = pred_sources[:, matches[0]]
    dst = sources[:, matches[1]]

    mask = np.logical_and(
        np.logical_and(pred_sources[0] >= -margin, pred_sources[0] <= width + margin),
        np.logical_and(pred_sources[1] >= -margin, pred_sources[1] <= height + margin)
    )
    pred_sources = pred_sources[:, mask]

    fig, ax = plt.subplots()
    ax.scatter(pred_sources[0], pred_sources[1], alpha=0.5)
    ax.scatter(sources[0], sources[1], alpha=0.5)
    ax.quiver(src[0], src[1], dst[0] - src[0], dst[1] - src[1], color="black", alpha=1.0, width=0.003, scale_units="xy", scale=1, angles="xy", headwidth=0, headlength=0, headaxislength=0)

    if margin > 0:
        _draw_rectangle(ax, 0, 0, width, height)

    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(-margin, height + margin)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def plot_night_sky(sources, vmags, width, height, crosshair=True, borderless=False, dpi=1000):
    mask = np.logical_and(
        np.logical_and(sources[0] >= 0, sources[0] <= width),
        np.logical_and(sources[1] >= 0, sources[1] <= height)
    )
    sources = sources[:, mask]
    vmags = vmags[mask]

    alpha = _apparent_magnitude_to_alpha(vmags)

    if borderless:
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
    else:
        fig, ax = plt.subplots()
    ax.set_facecolor("black")
    ax.scatter(sources[0], sources[1], s=0.5, color="white", alpha=alpha)
    if crosshair:
        _draw_crosshair(ax, width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_axes(ax)

    return fig


def save_night_sky(sources, vmags, width, height, path, crosshair=True):
    fig = plot_night_sky(sources, vmags, width, height, crosshair, True)
    fig.savefig(path)


def _apparent_magnitude_to_alpha(vmags, k=3):
    m = np.power(10, -0.4 * vmags)
    m = (1 + k) * m / (1 + k * m * m)
    return np.clip(m, 0, 1)


def _draw_rectangle(ax, x, y, width, height, color="black"):
    ax.plot([x, x + width], [y, y], color=color, alpha=0.5, linestyle="--")
    ax.plot([x + width, x + width], [y, y + height], color=color, alpha=0.5, linestyle="--")
    ax.plot([x + width, x], [y + height, y + height], color=color, alpha=0.5, linestyle="--")
    ax.plot([x, x], [y + height, y], color=color, alpha=0.5, linestyle="--")


def _draw_crosshair(ax, width, height, color="white"):
    ax.plot([0, width], [height / 2, height / 2], color=color, alpha=0.15, linestyle="--", linewidth=0.5)
    ax.plot([width / 2, width / 2], [0, height], color=color, alpha=0.15, linestyle="--", linewidth=0.5)
