# CPmethod

## Overview

This is the source code of the control points method for segmentation of short-axis cardiac MRI images.
The method supports the left and right ventricular endocardium segmentation.
Short description how to use the algorithm.

1. install the requirements from the requirements.txt (pip install -r requirements.txt)
2. set the path for your convinience in config.py (mostly where your data is located)
3. download the weights if you want to try the algorithm on a new dataset (or want a transfer learning) from the binaries in release v1.0
4. for training purpose the ground truth should be in the format prescribed by the algorithm, it is your responsibility to write the required transformation functions for your data set (as it is not known for me I can not provide a solution for that)
5. results folder will contain the results, two subfolder will be created for evaluations and training results separately. Each of these folders will contain the results in a folder with a unique name (contains the date as well)

## Additional information:

* the unit of the data is the Sample object which contains data about a patient, including the ground truth data too
* image_contour_3s in Sample is a dictionary with a hierarchical structure (slice -> heart_part -> phase (ES or ED) -> 'mask' or 'pred_mask' or 'ratio')
* 'mask' is the contour (ControlPoints class) or ROI (RoiBox class) even the name suggests something different

## Usage:

There are 3 files for running: train_cp.py, train_roi.py, eval.py. None of them requires additional arguments to launch due to the config.py.

## Resources:

The resource requirement is quite low, 2 GB GPU memory is enough and the training is quite fast, can be done within a day for a model (e.g: for the left side, roi + cp). However all of the models can be trained separately.

## Contact:

If you have any difficulty, please submit a new issue and I will help.
