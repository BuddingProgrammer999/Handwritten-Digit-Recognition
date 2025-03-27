import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import tkinter as tk
from tkinter import Canvas

model = load_model("mnist.h5")

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.canvas = Canvas(root, width=280, height=280, bg="black")
        self.canvas.pack()
        self.btn_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.btn_clear.pack(side="left")
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict_digit)
        self.btn_predict.pack(side="right")
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = np.zeros((280, 280), dtype=np.uint8)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+8, y+8, fill="white", outline="white")
        cv2.circle(self.image, (x, y), 8, 255, -1)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image.fill(0)

    def predict_digit(self):
        img_resized = cv2.resize(self.image, (28, 28))
        img_resized = img_resized / 255.0  
        img_resized = img_resized.reshape(1, 28, 28, 1) 
        prediction = model.predict(img_resized)
        digit = np.argmax(prediction)
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction")
        tk.Label(result_window, text=f"Recognized Digit: {digit}", font=("Arial", 20)).pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()