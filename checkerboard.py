from camera import Camera
from pathlib import Path
import cv2


camera_file = Path(__file__).parent / "config" / "XPRIZE.yaml"
image_dir = Path(__file__).parent / "data"

rows = 5
cols = 7
square_size = 0.0370
marker_size = 0.0184


if __name__ == "__main__":
    camera = Camera.from_file(camera_file)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    all_charuco_corners = []
    all_charuco_ids = []
    for path in image_dir.glob("checkerboard_*/*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if marker_corners is not None:
            image = cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

        if charuco_corners is not None:
            image = cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

            print("Found {} charuco corners in {}".format(len(charuco_corners), path.name))

    print("Initial camera matrix:")
    print(camera.camera_matrix)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        (image.shape[1], image.shape[0]),
        camera.camera_matrix,
        camera.distortion_coefficients,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)
