# Sentiment Analysis using CNN

This project demonstrates how to perform sentiment analysis on Twitter data using a Convolutional Neural Network (CNN). The model is built using TensorFlow and works on the `training.1600000.processed.noemoticon.csv` dataset, classifying tweets as positive or negative.

## Overview

The dataset contains tweets, each labeled as either positive (1) or negative (0). The goal is to build a CNN-based model to classify these tweets based on their sentiment. This approach avoids using transformers or complex architectures, relying instead on a more traditional machine learning approach.

## Dataset

The dataset used is `training.1600000.processed.noemoticon.csv`, which includes the following columns:

- `target`: The sentiment label (1 for positive, 0 for negative).
- `text`: The actual tweet text.
- Other metadata columns such as the timestamp and user details.

You can download the dataset from ([here](https://www.kaggle.com/datasets/kazanova/sentiment140)) or other sources providing it.

## Model Architecture

- **Preprocessing**: Text is preprocessed by converting to lowercase, removing stop words, and tokenizing the text.
- **CNN Model**: A Convolutional Neural Network is used for text classification. It learns spatial hierarchies of features in the text using convolutional layers and max-pooling operations.
- **Evaluation**: The model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/ahmdmohamedd/Sentiment-Analysis-CNN.git
    cd Sentiment-Analysis-CNN
    ```

2. Create a new virtual environment and activate it:
    ```bash
    conda create --name sentiment-analysis-cnn python=3.8
    conda activate sentiment-analysis-cnn
    ```

3. Download the dataset and place it in the project directory.

## Usage

1. Open the Jupyter notebook (`Sentiment_Analysis_CNN_Model.ipynb`).
2. Run the cells sequentially. The notebook will:
   - Load and preprocess the data.
   - Train a CNN model on the sentiment data.
   - Evaluate the model performance.
3. You will see the evaluation metrics (accuracy, loss) after training.

## Evaluation Metrics

- **Accuracy**: The overall classification performance of the model.
- **Precision, Recall, F1-Score**: These metrics help evaluate the model in terms of both the correct and incorrect classifications, especially when the data is imbalanced.

## Conclusion

This project provides a simple yet effective CNN-based model for sentiment analysis, specifically for Twitter data. By focusing on CNNs, the project showcases how convolutional architectures can be used for text classification tasks without the need for transformer models.
