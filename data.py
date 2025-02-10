from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy import units as u
from astropy.time import Time
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import yaml


@dataclass
class ImageMetadata:
    location: EarthLocation
    pressure: float
    temperature: float
    relative_humidity: float


@dataclass
class Image:
    width: int
    height: int
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
            pressure=self.metadata.pressure * u.hPa,
            temperature=self.metadata.temperature * u.deg_C,
            relative_humidity=self.metadata.relative_humidity,
            obswl=500 * u.nm,
        )

    def read(self):
        return cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    
    def to_local_frame(self, coords: SkyCoord) -> SkyCoord:
        c = coords.apply_space_motion(self.timestamp)
        c = SkyCoord(ra=c.ra, dec=c.dec)
        c = c.transform_to(self.local_frame)
        return c

    def to_local_atmo_frame(self, coords: SkyCoord) -> tuple[SkyCoord, np.ndarray]:
        c = self.to_local_frame(coords)
        mask = c.alt.deg > 10
        c = coords[mask].transform_to(self.local_atmo_frame)
        return c, np.nonzero(mask)[0]


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
            width=metadata_dict["width"],
            height=metadata_dict["height"],
            path=dataset_dir / image_dict["path"],
            timestamp=Time(image_dict["timestamp"]),
            metadata=metadata
        )
        images.append(image)

    return images
