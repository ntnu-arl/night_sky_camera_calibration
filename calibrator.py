from camera import Camera
from scipy.optimize import least_squares
import numpy as np


class Calibrator:
    def __init__(self, camera, orientation):
        self.camera = camera
        self.orientation = orientation
    
    def calibrate_orientation(self, catalog, sources):
        x0 = np.zeros(3)

        res = least_squares(
            self._orientation_objective,
            x0,
            args=(catalog, sources),
        )

        err = self._orientation_objective(res.x, catalog, sources)
        err = np.linalg.norm(err.reshape(2, -1), axis=0)

        self.orientation = self.orientation @ rodrigues(res.x)

        return err
    
    def calibrate_orientation_and_focal_length(self, catalog, sources):
        x0 = np.zeros(4)
        x0[3] = 1

        res = least_squares(
            self._orientation_and_focal_length_objective,
            x0,
            args=(catalog, sources),
        )

        err = self._orientation_and_focal_length_objective(res.x, catalog, sources)
        err = np.linalg.norm(err.reshape(2, -1), axis=0)

        self.camera = Camera(
            self.camera.width,
            self.camera.height,
            self.camera.camera_matrix @ np.diag([res.x[3], res.x[3], 1]),
            self.camera.distortion_coefficients
        )
        self.orientation = self.orientation @ rodrigues(res.x[:3])

        return err

    def calibrate_orientation_and_camera_matrix(self, catalog, sources):
        x0 = np.zeros(7)
        x0[3] = self.camera.camera_matrix[0, 0]
        x0[4] = self.camera.camera_matrix[1, 1]
        x0[5] = self.camera.camera_matrix[0, 2]
        x0[6] = self.camera.camera_matrix[1, 2]

        res = least_squares(
            self._orientation_and_camera_matrix_objective,
            x0,
            args=(catalog, sources),
        )

        err = self._orientation_and_camera_matrix_objective(res.x, catalog, sources)
        err = np.linalg.norm(err.reshape(2, -1), axis=0)

        self.camera = Camera(
            self.camera.width,
            self.camera.height,
            np.array([
                [res.x[3], 0, res.x[5]],
                [0, res.x[4], res.x[6]],
                [0, 0, 1]
            ]),
            self.camera.distortion_coefficients
        )
        self.orientation = self.orientation @ rodrigues(res.x[:3])

        return err

    def calibrate(self, catalog, sources):
        x0 = np.zeros(12)
        x0[3] = self.camera.camera_matrix[0, 0]
        x0[4] = self.camera.camera_matrix[1, 1]
        x0[5] = self.camera.camera_matrix[0, 2]
        x0[6] = self.camera.camera_matrix[1, 2]
        x0[7:] = self.camera.distortion_coefficients.flatten()

        res = least_squares(
            self._single_image_objective,
            x0,
            args=(catalog, sources),
        )

        err = self._single_image_objective(res.x, catalog, sources)
        err = np.linalg.norm(err.reshape(2, -1), axis=0)

        self.camera = Camera(
            self.camera.width,
            self.camera.height,
            np.array([
                [res.x[3], 0, res.x[5]],
                [0, res.x[4], res.x[6]],
                [0, 0, 1]
            ]),
            res.x[7:]
        )
        self.orientation = self.orientation @ rodrigues(res.x[:3])

        return err

    def _orientation_objective(self, x, catalog, sources):
        orientation = self.orientation @ rodrigues(x)
        pred_sources, idx = self.camera.project(catalog, orientation)
        idx = nearest_neighbour(sources, pred_sources)
        src = sources
        dst = pred_sources[:, idx]
        return (src - dst).flatten()

    def _orientation_and_focal_length_objective(self, x, catalog, sources):
        camera_matrix = self.camera.camera_matrix @ np.diag([x[3], x[3], 1])
        camera = Camera(
            self.camera.width,
            self.camera.height,
            camera_matrix,
            self.camera.distortion_coefficients
        )
        orientation = self.orientation @ rodrigues(x[:3])

        pred_sources, idx = camera.project(catalog, orientation)

        idx = nearest_neighbour(sources, pred_sources)
        src = sources
        dst = pred_sources[:, idx]

        return (src - dst).flatten()

    def _orientation_and_camera_matrix_objective(self, x, catalog, sources):
        camera = Camera(
            self.camera.width,
            self.camera.height,
            np.array([
                [x[3], 0, x[5]],
                [0, x[4], x[6]],
                [0, 0, 1]
            ]),
            self.camera.distortion_coefficients
        )
        orientation = self.orientation @ rodrigues(x[:3])

        pred_sources, idx = camera.project(catalog, orientation)

        idx = nearest_neighbour(sources, pred_sources)
        src = sources
        dst = pred_sources[:, idx]

        return (src - dst).flatten()

    def _single_image_objective(self, x, catalog, sources):
        camera = Camera(
            self.camera.width,
            self.camera.height,
            np.array([
                [x[3], 0, x[5]],
                [0, x[4], x[6]],
                [0, 0, 1]
            ]),
            x[7:]
        )
        orientation = self.orientation @ rodrigues(x[:3])

        pred_sources, idx = camera.project(catalog, orientation)

        idx = nearest_neighbour(sources, pred_sources)
        src = sources
        dst = pred_sources[:, idx]

        return (src - dst).flatten()


class MultiImageCalibrator:
    def __init__(self, camera, orientations):
        self.camera = camera
        self.orientations = orientations

    def calibrate(self, all_catalogs, all_sources):
        x0 = np.zeros(9 + 3 * len(all_catalogs))
        x0[0] = self.camera.camera_matrix[0, 0]
        x0[1] = self.camera.camera_matrix[1, 1]
        x0[2] = self.camera.camera_matrix[0, 2]
        x0[3] = self.camera.camera_matrix[1, 2]
        x0[4:9] = self.camera.distortion_coefficients.flatten()

        res = least_squares(
            self._multi_image_objective,
            x0,
            args=(all_catalogs, all_sources),
        )

        err = self._multi_image_objective(res.x, all_catalogs, all_sources)
        err = np.linalg.norm(err.reshape(2, -1), axis=0)

        self.camera = Camera(
            self.camera.width,
            self.camera.height,
            np.array([
                [res.x[0], 0, res.x[2]],
                [0, res.x[1], res.x[3]],
                [0, 0, 1]
            ]),
            res.x[4:9]
        )

        for i in range(len(self.orientations)):
            self.orientations[i] = self.orientations[i] @ rodrigues(res.x[9 + i * 3:9 + (i + 1) * 3])

        return err

    def _single_image_objective(self, x, catalog, sources, R0):
        camera = Camera(
            self.camera.width,
            self.camera.height,
            np.array([
                [x[3], 0, x[5]],
                [0, x[4], x[6]],
                [0, 0, 1]
            ]),
            x[7:]
        )
        orientation = R0 @ rodrigues(x[:3])

        pred_sources, idx = camera.project(catalog, orientation)

        idx = nearest_neighbour(sources, pred_sources)
        src = sources
        dst = pred_sources[:, idx]

        return (src - dst).flatten()

    def _multi_image_objective(self, x, all_catalogs, all_sources):
        all_errs = []
        for i in range(len(all_catalogs)):
            xi = np.concatenate([x[9 + i * 3:9 + (i + 1) * 3], x[:9]])
            all_errs.append(self._single_image_objective(xi, all_catalogs[i], all_sources[i], self.orientations[i]))
        return np.concatenate(all_errs)


def nearest_neighbour(src, dst):
    idx = np.ones(src.shape[1], dtype=int) * -1
    for i in range(src.shape[1]):
        dist = np.linalg.norm(src[:, i][:, None] - dst, axis=0)
        idx[i] = np.argmin(dist)
    return idx


def mutual_nearest_neighbour(src, dst):
    src_neighbours = nearest_neighbour(src, dst)
    dst_neighbours = nearest_neighbour(dst, src)
    src_idx = np.arange(src.shape[1])
    mask = src_idx == dst_neighbours[src_neighbours]
    return np.stack([src_idx[mask], src_neighbours[mask]])


def match_sources(pred_sources, sources, threshold=100):
    matches = mutual_nearest_neighbour(pred_sources, sources)
    src = pred_sources[:, matches[0]]
    dst = sources[:, matches[1]]
    dist = np.linalg.norm(src - dst, axis=0)
    mask = dist < threshold
    return matches[:, mask]


def skew_symmetric(v):
    return np.array([
        [  0, -v[2],  v[1]],
        [ v[2],   0, -v[0]],
        [-v[1],  v[0],   0]
    ])


def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    r = r / theta
    K = skew_symmetric(r)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return R
