# Night sky camera calibration
Calibrate cameras using one or more images of the night sky.

## Method
This program uses DAOFIND to detect stars in the images, and then matches the detected stars to a catalog. The discrepancy between the observed position of stars and their position according to the catalog is minimized in order to estimate the intrinsic and extrinsic parameters of the camera. A detected star and a star in the catalog are considered a match if they are mutual nearest neighbours, and the distance between them is less than a threshold. The program currently supports the following catalogs: BSC5, IRAS, PPM and SAO.

### Coarse calibration
The program supports using multiple images taken with different orientations. Calibration is performed in two steps. In the first step, referred to as coarse calibration, the orientation and focal length of the camera are estimated using individual images. The estimated parameters are then used as an initial guess for the second step, referred to as fine calibration, where all parameters are jointly estimated using multiple images. 

During coarse calibration, only the brightest stars are considered in order to decrease the density of stars, and thus increase the probability that matching stars are mutual nearest neighbours. Incorrect matches are still bound to occur, but as long as a sufficient number of matches are correct, the estimated parameters should be a better approximation of the true parameters than the initial guess. By iteratively estimating the parameters with a decreasing match threshold, the number of correct matches should increase, and the estimated parameters should converge towards the true parameters. In this step, only the orientation and focal length of the camera are estimated, as this was found to be more stable when the initial guess is poor. Due to the simplicity of this model, a fairy high match threshold may still result in correct matches being discarded near the edges of the image, where the distortion is greatest. Decreasing the threshold to intentionally discard matches near the edges seems to yield a better approximation of the orientation, particularly around the optical axis of the camera. The threshold can then be increased to include these matches in the final iteration.

### Fine calibration
The coarse calibration step results in a different estimate of the intrinsic parameters for each image, whereas in the fine calibration step the intrinsic parameters are assumed to be identical for all images, so the intrinsic parameters of an arbitrary image is chosen as the initial guess. At this point, the estimated parameters should be close to the true parameters, so the probability that matching stars are mutual nearest neighbours remains high as the density of stars is increased. In this step, all parameters are jointly estimated. The parameters are the orientation of the camera for each image, the camera matrix and the distortion coefficients. Like in the coarse calibration step, parameters are iteratively estimated with a decreasing match threshold.

## Usage
A dataset of images of the night sky captured with the XPRIZE camera can be downloaded by running the following command in the root of the repository:
```bash
git submodule update --init
```
Calibration can then be performed by running the following command:
```bash
python main.py
```
When running the program for the first time, the selected catalog (SAO by default) has to be downloaded, which may take a while. Downloaded catalogs are cached for subsequent runs. Detection results are also cached.
