# Weather Image Classification

A deep learning project for classifying weather conditions from images using Convolutional Neural Networks (CNNs).

---

## Overview

This project leverages a custom CNN model to automatically identify different weather conditions (such as sunny, cloudy, rainy, etc.) from image data. With the power of deep learning, the model learns visual patterns and features directly from weather images, enabling accurate and efficient weather classification.

---

## Features

- End-to-end workflow: data loading, preprocessing, model training, and evaluation
- Custom CNN architecture with multiple convolutional and dense layers
- Supports multi-class weather classification
- Achieves strong accuracy on test data
- Easily extendable to new weather categories or datasets

---

## Model Architecture

The model consists of:
- Multiple convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Dense (fully connected) layers for classification
- Final softmax output for multi-class prediction

Example model summary:
```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_9 (Conv2D)            (None, 123, 123, 32)      3,488
conv2d_10 (Conv2D)           (None, 118, 118, 32)      36,896
max_pooling2d_6 (MaxPooling2D) (None, 39, 39, 32)      0
conv2d_11 (Conv2D)           (None, 34, 34, 32)        36,896
max_pooling2d_7 (MaxPooling2D) (None, 11, 11, 32)      0
flatten_3 (Flatten)          (None, 3872)              0
dense_10 (Dense)             (None, 64)                247,872
dense_11 (Dense)             (None, 32)                2,080
dense_12 (Dense)             (None, 32)                1,056
dense_13 (Dense)             (None, 11)                363
=================================================================
Total params: 985,955
Trainable params: 328,651
Non-trainable params: 0
```

---

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow or Keras
- NumPy, Pandas, Matplotlib (for data manipulation and visualization)

### Usage

1. Prepare your dataset of weather images, organized by category.
2. Run the Jupyter notebook `Weather_image.ipynb` to train and evaluate the model.
3. Adjust model parameters or architecture as needed for your specific dataset.

---

## Results

- The model achieves an accuracy of approximately **72-76%** on the test set.
- Example evaluation output:
    ```
    accuracy: 0.7577 - loss: 2.7066
    ```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- TensorFlow/Keras for deep learning frameworks
- Open source weather image datasets

