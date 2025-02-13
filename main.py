from camera import Camera
from calibrator import Calibrator, MultiImageCalibrator, match_sources
from catalog import load_catalog
from data import load_images
from detector import Detector
from pathlib import Path
from plotting import plot_matches, save_night_sky
import matplotlib.pyplot as plt
import numpy as np
import warnings


warnings.filterwarnings("ignore", message="ERFA function")


uncalibrated_file = Path(__file__).parent / "config" / "uncalibrated.yaml"
calibrated_file = Path(__file__).parent / "config" / "night_sky.yaml"
image_dir = Path(__file__).parent / "data" / "night_sky"

orientation = np.array([
    [0, 1, 0], # Camera X-axis is East, Y-axis is North, and Z-axis is up
    [1, 0, 0], # Local frame is left-handed, so we need to flip the Y-axis
    [0, 0, 1]
])
detector_fwhm = 17.0
coarse_max_vmag = 5.0
coarse_detection_threshold = 12.0
coarse_matching_thresholds = [100, 75, 50, 25, 50]
fine_max_vmag = 6.5
fine_detection_threshold = 5.0
fine_matching_thresholds = [25, 5]


if __name__ == "__main__":
    camera = Camera.from_file(uncalibrated_file)

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

    print("Coarse calibration")
    detector = Detector(detector_fwhm, coarse_detection_threshold)
    orientations = []
    for image in images:
        calibrator = Calibrator(camera, orientation)
        local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < coarse_max_vmag])
        sources = detector.detect(image)
        for threshold in coarse_matching_thresholds:
            pred_sources, idx2 = calibrator.camera.project(local_catalog, calibrator.orientation)
            matches = match_sources(pred_sources, sources, threshold=threshold)
            err = calibrator.calibrate_orientation_and_focal_length(local_catalog[idx2[matches[0]]], sources[:, matches[1]])
            print(f"  Image: {image.path.name}, Threshold: {threshold}, Matches: {err.shape[0]}, Error: {err.mean().item():.2f}")
        orientations.append(calibrator.orientation)
    camera = calibrator.camera

    print("Fine calibration")
    detector = Detector(detector_fwhm, fine_detection_threshold)
    calibrator = MultiImageCalibrator(camera, orientations)
    for threshold in fine_matching_thresholds:
        all_catalogs = []
        all_sources = []
        for image, orientation in zip(images, calibrator.orientations):
            local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < fine_max_vmag])
            sources = detector.detect(image)
            pred_sources, idx2 = calibrator.camera.project(local_catalog, orientation)
            matches = match_sources(pred_sources, sources, threshold)
            all_catalogs.append(local_catalog[idx2[matches[0]]])
            all_sources.append(sources[:, matches[1]])
        err = calibrator.calibrate(all_catalogs, all_sources)
        print(f"  Threshold: {threshold}, Matches: {err.shape[0]}, Error: {err.mean().item():.2f}")
    camera = calibrator.camera
    orientations = calibrator.orientations

    camera.to_file(calibrated_file, error=err.mean().item())

    plt.hist(err, bins=50)
    plt.show()

    for image, orientation in zip(images, orientations):
        local_catalog, idx1 = image.to_local_atmo_frame(catalog[vmags < fine_max_vmag])
        sources = detector.detect(image)
        pred_sources, idx2 = camera.project(local_catalog, orientation)
        matches = match_sources(pred_sources, sources, threshold=threshold)

        fig = plot_matches(pred_sources, sources, matches, image.width, image.height)
        plt.show()

    # Avg error: 0.46
    # Avg distance from center of pixel to random point in pixel: 0.57
