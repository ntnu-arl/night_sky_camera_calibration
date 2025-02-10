from camera import Camera
from catalog import load_catalog
from data import load_images
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


camera_file = Path(__file__).parent / "config" / "XPRIZE.yaml"
image_dir = Path(__file__).parent / "data" / "night_sky"


R0 = np.array([
    [ 0, 1, 0],
    [-1, 0, 0],
    [ 0, 0, 1]
])


def draw_crosshair(ax, width, height, color="white"):
    ax.plot([0, width], [height / 2, height / 2], color=color, alpha=0.15, linestyle="--", linewidth=0.5)
    ax.plot([width / 2, width / 2], [0, height], color=color, alpha=0.15, linestyle="--", linewidth=0.5)


def apparent_magnitude_to_intensity(magnitude, k=2):
    magnitude = np.power(10, -0.4 * magnitude)
    magnitude = (1 + k) * magnitude / (1 + k * magnitude * magnitude)
    return np.clip(magnitude, 0, 1)


if __name__ == "__main__":
    camera = Camera.from_file(camera_file)
    catalog, vmags = load_catalog("SAO")

    images = load_images(image_dir)
    images = [
        images[6],
        images[25],
        images[44]
    ]

    for image in images:
        image_catalog, idx1 = image.to_local_atmo_frame(catalog)
        simualated_image, idx2 = camera.project(image_catalog, R0)
        simualated_image_vmags = vmags[idx1[idx2]]

        # print(image_catalog.shape)
        # print(idx.shape)

        alpha = apparent_magnitude_to_intensity(simualated_image_vmags, k=3)
        # reference = project(catalog, R0, image.camera_matrix, image.distortion_coefficients, image.width, image.height)

        dpi = 1000
        fig = plt.figure(figsize=(image.width/dpi, image.height/dpi), dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_facecolor("black")
        ax.scatter(simualated_image[0], simualated_image[1], s=0.5, color="white", alpha=alpha)
        draw_crosshair(ax, image.width, image.height)
        ax.set_xlim(0, image.width)
        ax.set_ylim(0, image.height)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        fig.add_axes(ax)
        fig.savefig(Path(__file__).parent / "output" / f"{image.path.stem}_sim.png")
