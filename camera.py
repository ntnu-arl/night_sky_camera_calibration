from astropy.coordinates import AltAz, SkyCoord
from pathlib import Path
import numpy as np
import yaml


class Camera:
    def __init__(self, camera_matrix, distortion_coefficients):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
    
    @classmethod
    def from_file(cls, path: Path):
        with open(path) as file:
            data = yaml.safe_load(file)
    
        if "camera_matrix" in data:
            camera_matrix = np.array(data["camera_matrix"])
        else:
            camera_matrix = _create_camera_matrix(
                data["focal_length"],
                (data["sensor_size"]["width"], data["sensor_size"]["height"]),
                (data["image_size"]["width"], data["image_size"]["height"])
            )

        if "distortion_coefficients" in data:
            distortion_coefficients = np.array(data["distortion_coefficients"])
        else:
            distortion_coefficients = np.zeros((5, 1))
        
        return cls(
            camera_matrix,
            distortion_coefficients
        )
    
    def to_file(self, path: Path):
        with open(path, "w") as file:
            yaml.dump({
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.distortion_coefficients.tolist()
            }, file)

    def to_camera_frame(self, coords: SkyCoord, orientation: np.ndarray):
        c = coords.cartesian.xyz.value.copy()
        c = orientation.T @ c
        return c

    def project(self, coords: SkyCoord, orientation: np.ndarray): #, width, height):
        idx = np.arange(len(coords))
        coords_camera = self.to_camera_frame(coords, orientation)

        mask = coords_camera[2] > 0
        idx = idx[mask]
        coords_camera = coords_camera[:, mask]

        x = coords_camera[0] / coords_camera[2]
        y = coords_camera[1] / coords_camera[2]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r = np.sqrt(x2 + y2)
        r2 = r * r
        r4 = r2 * r2
        r6 = r2 * r4
        d = 1 + self.distortion_coefficients[0] * r2 + self.distortion_coefficients[1] * r4 + self.distortion_coefficients[4] * r6
        x_d = x * d + 2 * self.distortion_coefficients[2] * xy + self.distortion_coefficients[3] * (r2 + 2 * x2)
        y_d = y * d + 2 * self.distortion_coefficients[3] * xy + self.distortion_coefficients[2] * (r2 + 2 * y2)
        x_d = x_d * self.camera_matrix[0, 0] + self.camera_matrix[0, 2]
        y_d = y_d * self.camera_matrix[1, 1] + self.camera_matrix[1, 2]

        return np.stack([x_d, y_d]), idx


def _create_camera_matrix(focal_length_mm, sensor_size_mm, image_size):
    fx = focal_length_mm / sensor_size_mm[0] * image_size[0]
    fy = focal_length_mm / sensor_size_mm[1] * image_size[1]
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
