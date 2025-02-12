from astropy.stats import sigma_clipped_stats
from data import Image
from pathlib import Path
from photutils.detection import DAOStarFinder
import hashlib
import numpy as np


class Detector:
    def __init__(self, fwhm: float, threshold: float, caching=True):
        self.fwhm = fwhm
        self.threshold = threshold
        self.caching = caching

    def detect(self, image: Image):
        if self.caching:
            cache_path = self._image_cache_path(image)
            if cache_path.exists():
                return np.load(cache_path)
        
        data = image.read()
        data = data.astype(np.float32) / 255
        mean, median, std = sigma_clipped_stats(data, sigma=self.threshold)

        daofind = DAOStarFinder(fwhm=self.fwhm, threshold=self.threshold * std)
        sources = daofind(data - median)
        sources = np.array([sources["xcentroid"], sources["ycentroid"]])

        if self.caching:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, sources)

        return sources

    @property
    def _cache_dir(self):
        return Path(__file__).parent / ".cache" / "night_sky_camera_calibration" / "detections"

    def _image_digest(self, image: Image):
        with open(image.path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256")

        digest.update(str(self.fwhm).encode())
        digest.update(str(self.threshold).encode())

        return digest.hexdigest()

    def _image_cache_path(self, image: Image):
        return self._cache_dir / f"{self._image_digest(image)}.npy"
