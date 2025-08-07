# Heart Rate Estimation from Contactless Fingerprint
This project aims to detect whether an image of a finger is real or fake by estimating the heart rate using remote photoplethysmography (rPPG) techniques. The rPPG signal is extracted from the fingertip images using a modified version of the PhyNet architecture, incorporating the Swin Transformer.
Overview
# The algorithm works as follows:

Read the input image and crop it to 128 x 128 pixels.
Read each image 100 times to create a video with a frame rate of 24 fps.
Generate an .h5 file for each video.
Use the modified PhyNet architecture to generate an rPPG block (spatio-temporal).
Calculate the Power Spectral Density (PSD) of the signal and apply a filter to estimate the heart rate.

# rPPG (Remote Photoplethysmography)
rPPG, or remote photoplethysmography, is a technique for measuring physiological signals such as heart rate and respiration rate by analyzing subtle changes in skin color that occur with each heartbeat.
Score Interpretation
The output of the algorithm provides a score ranging from 0 to 100 for each input image. A score closer to 0 indicates a higher probability that the image is fake, while a score between 50 and 100 indicates a higher probability that the image is real.
# Repository Structure
Model.py: Contains the implementation of the modified PhyNet architecture with the Swin Transformer.

functions.py: Contains helper functions for data preprocessing, video generation, and rPPG signal extraction.

loss.py: Defines the loss functions used for training the model.

power.py: Implements the PSD calculation and filtering for heart rate estimation.

swinvideo.py: Contains the implementation of the Swin Transformer module.

test.py: Provides scripts for testing the model on new data.

train.py: Includes the training loop and related functions.

utils_data.py: Contains data loading and preprocessing utilities.

utils_sig.py: Provides signal processing utilities for rPPG signal analysis.

requirement.txt: Lists the required Python packages and dependencies.

#Usage

Install the required Python packages listed in requirement.txt.

Prepare your dataset of fingertip images (both real and fake).

Run train.py to train the model on your dataset.

Use test.py to evaluate the trained model on new data.

The output will provide a score ranging from 0 to 100 for each input image. A score closer to 0 indicates a higher probability that the image is fake, while a score between 50 and 100 indicates a higher probability that the image is real.
