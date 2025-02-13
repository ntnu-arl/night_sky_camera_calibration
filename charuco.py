from camera import Camera
from pathlib import Path
import cv2
import multiprocessing


uncalibrated_file = Path(__file__).parent / "config" / "uncalibrated.yaml"
calibrated_file = Path(__file__).parent / "config" / "charuco.yaml"
image_dir = Path(__file__).parent / "data"

rows = 5
cols = 7
square_size = 0.0370
marker_size = 0.0184


def init_pool():
    global detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)


def detect(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
    return charuco_corners, charuco_ids


if __name__ == "__main__":
    uncalibrated = Camera.from_file(uncalibrated_file)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)

    pool = multiprocessing.Pool(initializer=init_pool)
    results = pool.map(detect, image_dir.glob("charuco_?/*.png"))
    pool.close()
    pool.join()

    all_charuco_corners, all_charuco_ids = zip(*results)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        (uncalibrated.width, uncalibrated.height),
        uncalibrated.camera_matrix,
        uncalibrated.distortion_coefficients,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    calibrated = Camera(
        uncalibrated.width,
        uncalibrated.height,
        camera_matrix,
        dist_coeffs
    )

    calibrated.to_file(calibrated_file, error=ret)
