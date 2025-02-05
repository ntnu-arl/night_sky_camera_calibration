from catalog import load_catalog
from data import load_images
from pathlib import Path


image_dir = Path(__file__).parent / "data" / "night_sky"


if __name__ == "__main__":
    images = load_images(image_dir)
    catalog, vmags = load_catalog("SAO")

    for i, image in enumerate(images):
        print(i, image)
        data = image.read()
        print(data.shape, data.dtype)
