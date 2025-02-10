from camera import Camera
from catalog import load_catalog
from data import load_images
from pathlib import Path
from plotting import save_night_sky
import numpy as np


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

        save_night_sky(
            simualated_image,
            simualated_image_vmags,
            image.width,
            image.height,
            Path(__file__).parent / "output" / f"{image.path.stem}_sim.png"
        )
