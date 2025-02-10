from camera import Camera
from catalog import load_catalog
from data import load_images
from detector import Detector
from pathlib import Path
from plotting import plot_night_sky, plot_sources, save_night_sky
import numpy as np

import matplotlib.pyplot as plt


camera_file = Path(__file__).parent / "config" / "XPRIZE.yaml"
image_dir = Path(__file__).parent / "data" / "night_sky"


# Camera X-axis is East, Y-axis is North, and Z-axis is up
# Local frame is left-handed, so we need to flip the Y-axis
R0 = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])


if __name__ == "__main__":
    camera = Camera.from_file(camera_file)
    catalog, vmags = load_catalog("SAO")
    mask = vmags < 6.5
    catalog = catalog[mask]
    vmags = vmags[mask]

    images = load_images(image_dir)
    images = [
        images[6],
        images[25],
        images[44]
    ]

    detector = Detector(fwhm=17.0, threshold=5.0)

    for image in images:
        local_catalog, idx1 = image.to_local_atmo_frame(catalog)
        pred_sources, idx2 = camera.project(local_catalog, R0)
        pred_sources_vmags = vmags[idx1[idx2]]

        sources = detector.detect(image)

        plot_sources(pred_sources, sources, image.width, image.height)
        # plot_night_sky(pred_sources, pred_sources_vmags, image.width, image.height)
        plt.show()

        # save_night_sky(
        #     pred_sources,
        #     pred_sources_vmags,
        #     image.width,
        #     image.height,
        #     Path(__file__).parent / "output" / f"{image.path.stem}_sim.png"
        # )
