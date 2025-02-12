from camera import Camera
from calibrator import Calibrator
from catalog import load_catalog
from data import load_images
from detector import Detector
from pathlib import Path
from plotting import plot_matches, plot_night_sky, plot_sources, save_night_sky
import matplotlib.pyplot as plt
import numpy as np


uncalibrated_file = Path(__file__).parent / "config" / "uncalibrated.yaml"
calibrated_file = Path(__file__).parent / "config" / "night_sky.yaml"
image_dir = Path(__file__).parent / "data" / "night_sky"


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


if __name__ == "__main__":
    camera = Camera.from_file(uncalibrated_file)
    orientation = np.array([
        [0, 1, 0], # Camera X-axis is East, Y-axis is North, and Z-axis is up
        [1, 0, 0], # Local frame is left-handed, so we need to flip the Y-axis
        [0, 0, 1]
    ])

    catalog, vmags = load_catalog("SAO")
    # mask = vmags < 6.5
    # catalog = catalog[mask]
    # vmags = vmags[mask]

    images = load_images(image_dir)
    images = [
        images[6],
        images[25],
        images[44]
    ]

    detector = Detector(fwhm=17.0, threshold=12.0)
    orientations = []
    for image in images:
        calibrator = Calibrator(camera, orientation)
        local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < 5])
        sources = detector.detect(image)
        for threshold in [100, 75, 50, 25, 50]:
            pred_sources, idx2 = calibrator.camera.project(local_catalog, calibrator.orientation)
            matches = match_sources(pred_sources, sources, threshold=threshold)
            err = calibrator.calibrate_orientation_and_focal_length(local_catalog[idx2[matches[0]]], sources[:, matches[1]])
            # print(f"Threshold: {threshold}, Matches: {err.shape[0]}, Error: {err.mean().item()}")
        orientations.append(calibrator.orientation)
    camera = calibrator.camera

    detector = Detector(fwhm=17.0, threshold=5.0)
    all_catalogs = []
    all_sources = []
    for image, orientation in zip(images, orientations):
        # calibrator = Calibrator(camera, orientation)
        local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < 6.5])
        sources = detector.detect(image)
        pred_sources, idx2 = camera.project(local_catalog, orientation)
        matches = match_sources(pred_sources, sources, 25)
        all_catalogs.append(local_catalog[idx2[matches[0]]])
        all_sources.append(sources[:, matches[1]])

        # err = calibrator.estimate_orientation_and_camera_matrix(local_catalog[idx2[matches[0]]], sources[:, matches[1]])
        # err = calibrator.calibrate(local_catalog[idx2[matches[0]]], sources[:, matches[1]])
        # print(f"Matches: {matches.shape[1]}, Error: {err.mean().item()}")

        # fx = calibrator.camera.camera_matrix[0, 0]
        # fy = calibrator.camera.camera_matrix[1, 1]
        # print((fx / 5320) * 14.58, (fy / 4600) * 12.60)

        # print(calibrator.camera.camera_matrix)
        # print(calibrator.camera.distortion_coefficients)

        # pred_sources, idx2 = calibrator.camera.project(local_catalog, calibrator.orientation)
        # matches = match_sources(pred_sources, sources, threshold=25)
        # fig = plot_matches(pred_sources, sources, matches, image.width, image.height)
        # plt.show()

    calibrator = Calibrator(camera, orientation)
    err, orientations = calibrator.calibrate_all(all_catalogs, all_sources, orientations)
    print("Error:", err.mean().item())

    calibrator.camera.to_file(calibrated_file, reprojection_error=err.mean().item())

    # plt.hist(err, bins=50)
    # plt.show()

    pred_sources, idx2 = calibrator.camera.project(local_catalog, orientations[-1])
    pred_sources_vmags = vmags[idx1[idx2]]
    matches = match_sources(pred_sources, sources, threshold=25)

    fig = plot_matches(pred_sources, sources, matches, image.width, image.height)
    # fig.savefig(Path(__file__).parent / "output" / "figures" / "matches_2.png", bbox_inches="tight", dpi=300)
    # save_night_sky(pred_sources, pred_sources_vmags, image.width, image.height, Path(__file__).parent / "output" / "figures" / "night_sky.png")
    plt.show()


    # save_night_sky(
    #     pred_sources,
    #     pred_sources_vmags,
    #     image.width,
    #     image.height,
    #     Path(__file__).parent / "output" / f"{image.path.stem}_sim.png"
    # )
