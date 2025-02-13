from camera import Camera
from calibrator import Calibrator, MultiImageCalibrator, match_sources
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

detector_fwhm = 17.0
coarse_max_vmag = 5.0
coarse_detection_threshold = 12.0
fine_max_vmag = 6.5
fine_detection_threshold = 5.0


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

    # Coarse calibration
    detector = Detector(detector_fwhm, coarse_detection_threshold)
    orientations = []
    for image in images:
        calibrator = Calibrator(camera, orientation)
        local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < coarse_max_vmag])
        sources = detector.detect(image)
        for threshold in [100, 75, 50, 25, 50]:
            pred_sources, idx2 = calibrator.camera.project(local_catalog, calibrator.orientation)
            matches = match_sources(pred_sources, sources, threshold=threshold)
            err = calibrator.calibrate_orientation_and_focal_length(local_catalog[idx2[matches[0]]], sources[:, matches[1]])
            print(f"Threshold: {threshold}, Matches: {err.shape[0]}, Error: {err.mean().item()}")
        orientations.append(calibrator.orientation)
    camera = calibrator.camera

    # Fine calibration
    detector = Detector(detector_fwhm, fine_detection_threshold)
    all_catalogs = []
    all_sources = []
    for image, orientation in zip(images, orientations):
        local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < fine_max_vmag])
        sources = detector.detect(image)
        pred_sources, idx2 = camera.project(local_catalog, orientation)
        matches = match_sources(pred_sources, sources, 25)
        all_catalogs.append(local_catalog[idx2[matches[0]]])
        all_sources.append(sources[:, matches[1]])  
    calibrator = MultiImageCalibrator(camera, orientations)
    err = calibrator.calibrate(all_catalogs, all_sources)
    camera = calibrator.camera
    orientations = calibrator.orientations

    # plt.hist(err, bins=50)
    # plt.show()

    for image, orientation in zip(images, orientations):
        local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < fine_max_vmag])
        sources = detector.detect(image)
        pred_sources, idx2 = camera.project(local_catalog, orientation)
        matches = match_sources(pred_sources, sources, threshold=25)

        src = sources[:, matches[1]]
        dst = pred_sources[:, matches[0]]
        err = np.linalg.norm(src - dst, axis=0)
        print("Matches:", err.shape[0], "Error:", err.mean().item())

        fig = plot_matches(pred_sources, sources, matches, image.width, image.height)
        plt.show()

    # save_night_sky(
    #     pred_sources,
    #     pred_sources_vmags,
    #     image.width,
    #     image.height,
    #     Path(__file__).parent / "output" / f"{image.path.stem}_sim.png"
    # )
