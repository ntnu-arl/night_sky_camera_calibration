from astropy.coordinates import AltAz, EarthLocation
from astropy import units as u
from astropy.time import Time
from dataclasses import dataclass
from pathlib import Path
import cv2
import yaml


@dataclass
class ImageMetadata:
    location: EarthLocation
    pressure: float
    temperature: float
    relative_humidity: float


@dataclass
class Image:
    path: Path
    timestamp: Time
    metadata: ImageMetadata

    @property
    def local_frame(self):
        return AltAz(
            location=self.metadata.location,
            obstime=self.timestamp,
        )

    @property
    def local_atmo_frame(self):
        return AltAz(
            location=self.metadata.location,
            obstime=self.timestamp,
            pressure=self.metadata.pressure,
            temperature=self.metadata.temperature,
            relative_humidity=self.metadata.relative_humidity,
            obswl=500 * u.nm,
        )

    def read(self):
        return cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)


def load_images(dataset_dir: Path):
    with open(dataset_dir / "metadata.yaml") as file:
        metadata_dict = yaml.safe_load(file)

    metadata = ImageMetadata(
        location=EarthLocation.from_geodetic(
            lon=metadata_dict["location"]["longitude"],
            lat=metadata_dict["location"]["latitude"],
            height=metadata_dict["location"]["elevation"]
        ),
        pressure=metadata_dict["conditions"]["pressure"],
        temperature=metadata_dict["conditions"]["temperature"],
        relative_humidity=metadata_dict["conditions"]["relative_humidity"]
    )

    images: list[Image] = []
    for image_dict in metadata_dict["images"]:
        image = Image(
            path=dataset_dir / image_dict["path"],
            timestamp=Time(image_dict["timestamp"]),
            metadata=metadata
        )
        images.append(image)

    return images
