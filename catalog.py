from abc import ABC, abstractmethod
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from struct import Struct
from tqdm import tqdm
import numpy as np
import requests


class Catalog(ABC):
    uri: str
    filename: str

    @property
    def cache_dir(self):
        return Path(__file__).parent / ".cache" / "night_sky_camera_calibration" / "catalogs"

    @property
    def path(self):
        return self.cache_dir / self.filename

    def download(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with requests.get(self.uri, stream=True) as response:
            response.raise_for_status()
            filesize = int(response.headers.get("content-length", 0))
            with open(self.path, "wb") as file, tqdm(
                desc=self.filename,
                total=filesize,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    bar.update(len(chunk))

    def load(self):
        if not self.path.exists():
            self.download()
        return self.read()

    @abstractmethod
    def read(self) -> tuple[SkyCoord, np.ndarray]:
        pass


class BSC5(Catalog):
    uri = "http://tdc-www.harvard.edu/catalogs/BSC5"
    filename = "BSC5"

    header_type = Struct("<7i")
    entry_type = Struct("<f2d2ch2f")

    def read(self):
        with open(self.path, "rb") as file:
            header = self.header_type.unpack(file.read(self.header_type.size))
            epoch = "J2000" if header[2] < 0 else "B1950"
            size = abs(header[2])
            ra = np.zeros(size)
            dec = np.zeros(size)
            pm_ra = np.zeros(size)
            pm_dec = np.zeros(size)
            vmag = np.zeros(size)
            for i in range(size):
                entry = self.entry_type.unpack(file.read(self.entry_type.size))
                ra[i] = entry[1]
                dec[i] = entry[2]
                pm_ra[i] = entry[6]
                pm_dec[i] = entry[7]
                vmag[i] = entry[5] / 100
        return (
            SkyCoord(
                ra=ra * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                dec=dec * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                pm_ra_cosdec=pm_ra * np.cos(dec) * u.rad/u.yr, # pyright: ignore[reportAttributeAccessIssue]
                pm_dec=pm_dec * u.rad/u.yr, # pyright: ignore[reportAttributeAccessIssue]
                obstime=epoch
            ),
            vmag
        )


class PPM(Catalog):
    uri = "http://tdc-www.harvard.edu/catalogs/PPM"
    filename = "PPM"

    header_type = Struct(">7i")
    entry_type = Struct(">2d2ch2f")

    def read(self):
        with open(self.path, "rb") as file:
            header = self.header_type.unpack(file.read(self.header_type.size))
            size = header[2]
            ra = np.zeros(size)
            dec = np.zeros(size)
            pm_ra = np.zeros(size)
            pm_dec = np.zeros(size)
            vmag = np.zeros(size)
            for i in range(size):
                entry = self.entry_type.unpack(file.read(self.entry_type.size))
                ra[i] = entry[0]
                dec[i] = entry[1]
                pm_ra[i] = entry[5]
                pm_dec[i] = entry[6]
                vmag[i] = entry[4] / 100
        return (
            SkyCoord(
                ra=ra * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                dec=dec * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                pm_ra_cosdec=pm_ra * np.cos(dec) * u.rad/u.yr, # pyright: ignore[reportAttributeAccessIssue]
                pm_dec=pm_dec * u.rad/u.yr, # pyright: ignore[reportAttributeAccessIssue]
                obstime="B1950"
            ),
            vmag
        )


class IRAS(Catalog):
    uri = "http://tdc-www.harvard.edu/catalogs/IRAS"
    filename = "IRAS"

    header_type = Struct(">7i")
    entry_type = Struct(">f2d2c4h")

    def read(self): # pyright: ignore[reportIncompatibleMethodOverride] FIXME
        with open(self.path, "rb") as file:
            header = self.header_type.unpack(file.read(self.header_type.size))
            size = header[2]
            ra = np.zeros(size)
            dec = np.zeros(size)
            for i in range(size):
                entry = self.entry_type.unpack(file.read(self.entry_type.size))
                ra[i] = entry[1]
                dec[i] = entry[2]
        return (
            SkyCoord(
                ra=ra * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                dec=dec * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                obstime="B1950"
            ),
            None
        )


class SAO(Catalog):
    uri = "http://tdc-www.harvard.edu/catalogs/SAO"
    filename = "SAO"

    header_type = Struct(">7i")
    entry_type = Struct(">2d2ch2f")

    def read(self):
        with open(self.path, "rb") as file:
            header = self.header_type.unpack(file.read(self.header_type.size))
            epoch = "J2000" if header[2] < 0 else "B1950"
            size = abs(header[2])
            ra = np.zeros(size)
            dec = np.zeros(size)
            pm_ra = np.zeros(size)
            pm_dec = np.zeros(size)
            vmag = np.zeros(size)
            for i in range(size):
                entry = self.entry_type.unpack(file.read(self.entry_type.size))
                ra[i] = entry[0]
                dec[i] = entry[1]
                pm_ra[i] = entry[5]
                pm_dec[i] = entry[6]
                vmag[i] = entry[4] / 100
        return (
            SkyCoord(
                ra=ra * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                dec=dec * u.rad, # pyright: ignore[reportAttributeAccessIssue]
                pm_ra_cosdec=pm_ra * np.cos(dec) * u.rad/u.yr, # pyright: ignore[reportAttributeAccessIssue]
                pm_dec=pm_dec * u.rad/u.yr, # pyright: ignore[reportAttributeAccessIssue]
                obstime=epoch
            ),
            vmag
        )


catalogs: dict[str, type[Catalog]] = {
    "BSC5": BSC5,
    "IRAS": IRAS,
    "PPM": PPM,
    "SAO": SAO
}


def load_catalog(name: str):
    catalog = catalogs[name]()
    return catalog.load()
