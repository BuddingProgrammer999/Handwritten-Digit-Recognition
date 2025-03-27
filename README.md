# Handwritten Digit Recognition using CNN

## Overview
This project implements a **Handwritten Digit Recognition** system using a **Convolutional Neural Network (CNN)**. The system consists of two main parts:
1. **Model Training (program.py)** - A CNN model is trained on the **MNIST** dataset.
2. **Digit Recognition GUI (practiceground.py)** - A graphical interface allows users to draw digits, which are then classified using the trained model.

---

## Features
- **CNN-based Model** for digit classification.
- **Tkinter GUI** for drawing and recognizing digits.
- **Uses OpenCV for image processing.**
- **Model is saved in mnist.h5** for later use.

---

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- Tkinter
- NumPy

---

## Dataset Used
- **MNIST Dataset**: A collection of 60,000 training and 10,000 testing images of handwritten digits (0-9).

---

## File Descriptions
### **1. Model Training (program.py)**
This script trains a **CNN** on the MNIST dataset and saves the model (`mnist.h5`).
#### **Key Steps:**
- Loads MNIST dataset.
- Preprocesses the images (resizing, normalizing, reshaping).
- Builds a CNN with:
  - Convolutional layers for feature extraction.
  - MaxPooling layers for dimensionality reduction.
  - Dense layers for classification.
- Trains the model and evaluates performance.
- Saves the trained model for later use.

#### **Run the Training Script:**
```bash
python train_model.py
```

---

### **2. Handwritten Digit Recognition GUI (practiceground.py)**
This script provides a graphical interface where users can draw a digit, and the model predicts the digit.
#### **Key Features:**
- **Tkinter Canvas**: Users can draw digits.
- **Clear Button**: Reset the canvas.
- **Predict Button**: Processes the image and predicts the digit.
- **Uses OpenCV** to process the drawn image before passing it to the model.

#### **Run the GUI Application:**
```bash
python main.py
```

---

## How It Works
1. **Train the model** using `program.py`. This saves the trained model as `mnist.h5`.
2. **Run the GUI** using `practiceground.py`.
3. **Draw a digit** in the Tkinter window.
4. Click the "Predict" button to classify the digit.
5. The predicted digit is displayed in a new window.

---

## Requirements
Install dependencies using:
```bash
pip install numpy opencv-python tensorflow keras
```

---

## Future Enhancements
- Improve model accuracy with more training.
- Add support for different handwriting styles.
- Deploy as a web-based application.

---

## Credits
- **Dataset**: MNIST (by Yann LeCun)
- **Libraries**: TensorFlow, OpenCV, Tkinter

