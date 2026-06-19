
LoadingRoomv3 (P.H.P) - v8 2026-06-12 2:31pm
==============================

This dataset was exported via roboflow.com on June 17, 2026 at 2:45 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1482 images.
LoadingRoomv3 are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 320x320 (Fit within)
* Auto-contrast via contrast stretching

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random brigthness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 2 pixels
* Salt and pepper noise was applied to 1 percent of pixels


