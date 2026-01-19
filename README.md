William Shaw
Image Classification Pipeline in Python and Pytorch
  
This repository implements an end-to-end image classification pipeline using a range of machine learning techniques. 
The project covers the full workflow from data loading and preprocessing through feature extraction, custom classifier design, 
model training, and evaluation, with an emphasis on clean structure and reproducibility.

The implementation uses the CIFAR-10 dataset, with 5,000 training images and 1,000 testing 
images, and is structured to allow different feature representations and classification 
approaches to be swapped and compared

What this project demonstrates:
- Design of a modular machine learning pipeline for supervised image classification
- Feature extraction and representation engineering for image data
- Implementation of custom classification algorithms on top of extracted features
- Training and evaluation of multiple modeling approaches
- Comparative analysis of model performance across different techniques

This project was developed in an academic setting as part of an artificial intelligence course
Below are some instructions for running the different modules of the project:
- This project uses a standard venv virtual environment with python version 3.12.3
- In order to run all the tests and get the evaluation table run the TestAll.py file
- Any of the other test files can be run but they only produce confusion matrices on their own
- In order to reproduce the feature vectors the Proprocessing.py file can be run (Saved feature vectors were included in the submission as an example)
- Any of the train files can be run to train any of the 4 categories of models (All variations of a given category are trained at once)
- Model architectures are stored in separate files from training and testing
- Naive bayes is never stored as a model as training time is negligible
