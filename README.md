# Night sky camera calibration
Calibrate cameras using one or more images of the night sky.

## Method
This program uses DAOFIND to detect stars in the images, and then matches the detected stars to a catalog. The discrepancy between the observed position of stars and their position according to the catalog is minimized in order to estimate the intrinsic and extrinsic parameters of the camera. A detected star and a star in the catalog are considered a match if they are mutual nearest neighbours, and the distance between them is less than a threshold. The program currently supports the following catalogs: BSC5, IRAS, PPM and SAO.

## Usage
A dataset of images of the night sky captured with the XPRIZE camera can be downloaded by running the following command in the root of the repository:
```bash
curl https://mikkelst.folk.ntnu.no/night_sky_camera_calibration_data.tar.gz | tar xzf -
```
Calibration can then be performed by running the following command:
```bash
python main.py
```
When running the program for the first time, the selected catalog (SAO by default) has to downloaded, which may take a while. Downloaded catalogs are cached for subsequent runs. Detection results are also cached.
