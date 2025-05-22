import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os
import time

# Load models
classifier = load_model('models/face_recognition_model.h5')
feature_extractor = load_model('models/vgg16_feature_extractor.h5')

# Load class names
with open('models/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f]

# GUI setup
root = tk.Tk()
root.title("🧠 Facial Recognition Classifier")
root.geometry("700x750")
root.configure(bg="#f4f4f4")

# Optional dark mode colors (uncomment if needed)
# bg_color = "#1e1e1e"
# fg_color = "#ffffff"
# btn_color = "#3a86ff"

# Light mode colors
bg_color = "#f4f4f4"
fg_color = "#222222"
btn_color = "#0077cc"

uploaded_image_path = None  # Global image path

# Header
header = Label(root, text="Face Recognition System", font=("Segoe UI", 24, "bold"),
               bg=bg_color, fg=btn_color)
header.pack(pady=20)

# Frame for image and results
main_frame = Frame(root, bg=bg_color)
main_frame.pack()

img_label = Label(main_frame, bg=bg_color)
img_label.grid(row=0, column=0, padx=10, pady=10)

# Prediction output
result_label = Label(main_frame, text="", font=("Segoe UI", 16), bg=bg_color, fg=fg_color)
confidence_label = Label(main_frame, text="", font=("Segoe UI", 14), bg=bg_color, fg=fg_color)
result_label.grid(row=1, column=0, pady=10)
confidence_label.grid(row=2, column=0)

# Functions
def predict_image():
    global uploaded_image_path
    if not uploaded_image_path:
        result_label.config(text="⚠️ Please upload an image first.")
        return

    img = keras_image.load_img(uploaded_image_path, target_size=(128, 128))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    features = feature_extractor.predict(img_array)
    features_flat = features.reshape(1, -1).astype(np.float32)

    start = time.time()
    preds = classifier.predict(features_flat)[0]
    elapsed = round(time.time() - start, 2)

    top_idx = np.argmax(preds)
    label = class_names[top_idx]
    confidence = round(preds[top_idx] * 100, 2)

    result_label.config(text=f"✅ Prediction: {label}")
    confidence_label.config(text=f"🎯 Confidence: {confidence}%   🕒 {elapsed}s")

def upload_image():
    global uploaded_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        uploaded_image_path = file_path
        img = Image.open(file_path).resize((300, 300))
        tk_img = ImageTk.PhotoImage(img)
        img_label.config(image=tk_img)
        img_label.image = tk_img
        result_label.config(text="")
        confidence_label.config(text="")

# Styled buttons
btn_font = ("Segoe UI", 12, "bold")upload_btn = Button(root, text="📤 Upload Image", command=upload_image,
                    font=btn_font, bg=btn_color, fg="white", width=20, height=2)
predict_btn = Button(root, text="🔮 Predict", command=predict_image,
                     font=btn_font, bg="#28a745", fg="white", width=20, height=2)

upload_btn.pack(pady=10)
predict_btn.pack(pady=10)

# Footer
footer = Label(root, text="© 2025 Facial Recognition System", font=("Segoe UI", 10),
               bg=bg_color, fg="#888888")
footer.pack(pady=20)

root.mainloop()
