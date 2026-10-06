
# Breast Ultrasound Lesion Segmentation Using Attention U-Net

## Project Overview

This project presents an Attention U-Net based deep learning system for automatic lesion segmentation in breast ultrasound images.

The system applies image preprocessing techniques to improve ultrasound image quality and uses an Attention U-Net model to generate a binary lesion segmentation mask.

> This project is an academic/research prototype for lesion segmentation and is not intended to be used as an independent clinical diagnostic system.

## Dataset

The project uses the Breast Ultrasound Images (BUSI) dataset.

Dataset used in this project:

- Total images: 629
- Benign: 420
- Malignant: 209
- Segmentation masks: 629

The dataset is not included in this repository.

## Methodology

The workflow consists of:

1. Dataset preparation
2. Image and mask pairing
3. Image preprocessing
4. Image resizing and normalization
5. Attention U-Net training
6. Lesion mask prediction
7. Model evaluation
8. Gradio web interface

## Image Preprocessing

The following preprocessing operations are applied:

- Median Filtering
- Sharpening
- Gamma Correction
- CLAHE
- Resizing to 256 × 256
- Pixel normalization to 0–1

## Model

The project uses an Attention U-Net architecture consisting of:

- Encoder
- Bottleneck
- Decoder
- Skip connections
- Attention gates
- Sigmoid segmentation output

The model is trained using a combined Binary Cross-Entropy and Dice loss.

## Dataset Split

The dataset was divided into:

- Training: 503 images
- Validation: 63 images
- Testing: 63 images

## Evaluation Results

| Metric | Score |
|---|---:|
| Dice / F1-score | 0.7344 |
| IoU | 0.5803 |
| Precision | 0.7994 |
| Recall | 0.6793 |
| Accuracy | 0.9469 |

## Web Interface

A Gradio-based web interface allows the user to upload a breast ultrasound image and obtain:

- Processed ultrasound image
- Predicted lesion segmentation mask

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Gradio
- Google Colab

## Repository Contents

- `breast_ultrasound_segmentation.ipynb` – Complete project notebook
- `app.py` – Gradio application
- `requirements.txt` – Python dependencies
- `LICENSE` – MIT License

## License

This project is licensed under the MIT License.

## Disclaimer

This project is developed for academic and research purposes. The segmentation results should not be considered a medical diagnosis or a replacement for professional clinical assessment.
