# Lung Disease Detection
Lung Disease Using Deep Learning
This project aims to build a lung disease detection model using the VGG16 convolutional neural network architecture. The model will be trained on a dataset of chest X-ray images to classify them as either normal or showing signs of pneumonia.

## Dataset
The dataset used in this project is the Chest X-Ray Images (Pneumonia) dataset from Kaggle. You can find the dataset here: Kaggle Dataset - Chest X-Ray Images (Pneumonia)

## Installation
To run this code, you will need to have the required dependencies installed. The necessary libraries include NumPy, pandas, Matplotlib, Seaborn, OpenCV, TensorFlow, and Keras

## Usage
Clone this repository to your local machine.
Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle and place it in the appropriate directory as specified in the code.
Open the Jupyter Notebook lung-disease.ipynb.
Run the notebook cells sequentially to execute the code step-by-step.

## Methodology
Data Preprocessing: The code reads the dataset, splits it into train and validation sets, and applies data augmentation to the training set using the ImageDataGenerator from TensorFlow.
Model Architecture: The VGG16 model is loaded, and some additional layers are added to fine-tune it for the lung disease detection task.
Model Training: The modified VGG16 model is trained on the training set using the Adam optimizer and binary cross-entropy loss.
Model Evaluation: The model is evaluated on the validation set to monitor its performance.

## Results
After running the code, the model's performance metrics will be printed, including validation loss and accuracy.
