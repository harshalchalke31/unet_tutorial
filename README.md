# UNet Implementation for Image Segmentation on Carvana Dataset

This repository contains the implementation of a **UNet** model for **image segmentation**. The model is trained on the **Carvana Image Masking Dataset**, where the task is to segment the car in the images. This project uses **PyTorch** for model building and training, with essential preprocessing and utilities designed to handle the dataset efficiently.

## Files in This Repository

1. **`unet.py`**: Contains the UNet model implementation.
2. **`preprocessing.py`**: Handles image preprocessing, including resizing, normalizing, and preparing the data for training.
3. **`utils.py`**: Utility functions for loading datasets, saving models, and calculating metrics such as the Intersection over Union (IoU).
4. **`main.ipynb`**: The main notebook that ties together the model, training loop, and evaluation.
5. **`hyperparams.json`**: Configuration file containing hyperparameters for the training process.

## Project Overview

This project focuses on building a **UNet-based model** for segmenting cars in images. The **UNet** architecture is widely used in biomedical and segmentation tasks due to its encoder-decoder structure, allowing for precise localization and segmentation of objects within an image.

### Dataset

The dataset used is the **Carvana Image Masking Dataset**, where the goal is to predict a mask for each car image, segmenting the car from the background. The dataset is available on **Kaggle**, and you can download it using the following link:

- [Carvana Image Masking Dataset](https://www.kaggle.com/c/carvana-image-masking-challenge/data)

This repository assumes the following folder structure for training and validation data:

- **Training images**: `./data/train_images/`
- **Training masks**: `./data/train_masks/`
- **Validation images**: `./data/val_images/`
- **Validation masks**: `./data/val_masks/`

### Architecture Overview

The **UNet** model follows a typical encoder-decoder structure:

1. **Encoder**: 
   - The encoder uses convolutional blocks with increasing feature maps and down-sampling through max-pooling layers.
   - This part of the network captures the context of the image.

2. **Bottleneck**:
   - The bottleneck serves as the transition between the encoder and decoder, capturing the most abstract representation of the input.

3. **Decoder**:
   - The decoder uses transposed convolutions to up-sample the feature maps, combining them with the corresponding feature maps from the encoder.
   - This part of the network helps in accurate spatial localization.

4. **Final Output**:
   - The output layer applies a **1x1 convolution** to produce a mask with the same spatial dimensions as the input image.

### Hyperparameters

The `hyperparams.json` file contains the configuration used for training:
- **Learning Rate**: `0.0001`
- **Device**: `cuda`
- **Batch Size**: `16`
- **Number of Epochs**: `3`
- **Number of Workers**: `2`
- **Image Dimensions**: `160 x 240`
- **Pin Memory**: `True`
- **Model Checkpoint Loading**: `True` (pre-trained model can be loaded)
- **Training Image Directory**: `./data/train_images/`
- **Training Mask Directory**: `./data/train_masks/`
- **Validation Image Directory**: `./data/val_images/`
- **Validation Mask Directory**: `./data/val_masks/`

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/unet-carvana-segmentation.git
   cd unet-carvana-segmentation

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the main notebook for training and validation:
   ```bash
   jupyter notebook main.ipynb

4. Alternatively, you can run the script by setting your configurations in hyperparams.json and executing the training:
   ```bash
   python train.py

## Preprocessing

- **Image Resizing**: All images and masks are resized to the dimensions `160x240` before being passed to the model for training and validation. This ensures consistency in input size.
- **Normalization**: Images are normalized to enhance convergence during training.
- **Augmentation**: Basic data augmentation techniques such as flipping and scaling may be added to improve generalization.

## Evaluation Metrics

We evaluate the model's performance using:
- **Intersection over Union (IoU)**: Calculated during training and validation to measure the overlap between predicted and ground truth masks.
- **Dice Coefficient**: Another metric used to measure segmentation quality, especially when dealing with imbalanced datasets.

## Concepts Covered

- **Image Segmentation**: The goal is to classify each pixel in the image as belonging to a car or the background.
- **UNet Architecture**: A powerful model for image segmentation tasks with encoder-decoder design.
- **Data Preprocessing**: Techniques to normalize and resize images, ensuring consistent input to the model.
- **Evaluation with IoU**: Using the Intersection over Union metric to evaluate model performance in segmentation tasks.

## Conclusion

This project provides a **beginner-friendly** yet **comprehensive** approach to **image segmentation** using the UNet model. By using the **Carvana dataset**, you can apply this model to real-world image segmentation tasks and expand it further by adding more layers or fine-tuning hyperparameters.

---

### Future Work

- **Data Augmentation**: Implement more advanced augmentation techniques such as rotation, contrast adjustments, etc., to improve model robustness.
- **Fine-tuning**: Explore hyperparameter tuning and model architecture changes to improve performance.
- **Post-Processing**: Apply techniques such as conditional random fields (CRFs) to refine mask predictions.

Feel free to contribute to this project or raise issues if you encounter any problems!

---

