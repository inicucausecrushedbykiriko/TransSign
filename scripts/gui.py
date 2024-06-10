from tkinter import *
import numpy as np
from PIL import Image, ImageGrab, ImageTk
import torch
import torch.nn as nn
import os

# Define the model class
class SignLanguageModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SignLanguageModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.softmax(self.layer2(out))
        return out

def predict_sign(img, model_path):
    input_size = 21 * 2  # 21 landmarks with x and y coordinates
    model = SignLanguageModel(input_size, 100, 10)  # Adjust the input size
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    img = img.resize((28, 28)).convert('L')
    img = np.array(img).reshape(1, 28*28)
    img = torch.tensor(img, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, axis=1).item()
    return prediction

def on_clear():
    canvas.delete("all")

def on_draw(event):
    global lastx, lasty
    x, y = event.x, event.y
    canvas.create_line((lastx, lasty, x, y), width=5, fill='black', capstyle=ROUND, smooth=True)
    lastx, lasty = x, y

def on_activate(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def on_predict():
    widget = canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    img = ImageGrab.grab().crop((x, y, x1, y1))
    sign = predict_sign(img, model_path)
    result_label.config(text=f'Sign = {sign}')
    show_translation_image(sign)

def show_translation_image(sign):
    translation_image_path = translation_dict.get(sign, "")
    if translation_image_path:
        img = ImageTk.PhotoImage(file=translation_image_path)
        translation_label.config(image=img)
        translation_label.image = img

def switch_mode():
    global model_path
    mode = mode_var.get()
    if mode == "ASL":
        model_path = 'models/asl_sign_language_model.pth'
    else:
        model_path = 'models/csl_sign_language_model.pth'
    result_label.config(text=f"Switched to {mode} mode")

window = Tk()
window.title("Sign Language Recognition")

mode_var = StringVar(value="ASL")
mode_menu = OptionMenu(window, mode_var, "ASL", "CSL", command=lambda _: switch_mode())
mode_menu.grid(row=0, column=0, pady=2)

canvas = Canvas(window, width=300, height=300, bg='white')
canvas.grid(row=1, column=0, pady=2, sticky=W, columnspan=2)

canvas.bind('<B1-Motion>', on_draw)
canvas.bind('<Button-1>', on_activate)

clear_button = Button(window, text='Clear', command=on_clear)
clear_button.grid(row=2, column=0, pady=2)

predict_button = Button(window, text='Predict', command=on_predict)
predict_button.grid(row=2, column=1, pady=2)

result_label = Label(window, text='Draw a sign and click Predict')
result_label.grid(row=3, column=0, columnspan=2, pady=2)

translation_label = Label(window)
translation_label.grid(row=4, column=0, columnspan=2, pady=2)

# Set the initial model path
model_path = 'models/asl_sign_language_model.pth'

# Dictionary to map ASL digits to CSL image paths
translation_dict = {
    1: 'data/sign_images/csl/images/1.png',
}

window.mainloop()
