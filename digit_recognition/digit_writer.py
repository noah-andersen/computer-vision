import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor
import sys
import os

# Create a GUI for drawing a digit
class HandwrittenDigitGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Drawer")
        self.setup_canvas()
        self.setup_buttons()
        self.setup_image()
        self.setup_prediction_label()

    def setup_canvas(self):
        self.canvas = Canvas(self.master, width=280, height=280, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

    def setup_buttons(self):
        self.button_save = Button(self.master, text="Save", command=self.save_and_predict)
        self.button_save.pack(side=tk.LEFT)
        self.button_exit = Button(self.master, text="Exit", command=self.exit_program)
        self.button_exit.pack(side=tk.RIGHT)

    def setup_image(self):
        self.image = Image.new("L", (280, 280), "black")
        self.draw_tool = ImageDraw.Draw(self.image)

    def setup_prediction_label(self):
        self.prediction_label = tk.Label(self.master, text="")
        self.prediction_label.pack()

    def draw(self, event):
        x, y = event.x, event.y
        r = 6
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white")
        
        # Create a new drawing context for each draw action
        draw_tool = ImageDraw.Draw(self.image)
        draw_tool.ellipse([x-r, y-r, x+r, y+r], fill="white")


    def save_and_predict(self):
        filename = "handwritten_digit.png"
        full_path = os.path.abspath(filename)
        
        if os.path.exists(full_path):
            os.remove(filename)

        img_resized = self.image.resize((28, 28))
        img_resized.save(filename)

        self.canvas.delete("all")

        digit_recognizer = CNN().cuda()
        digit_recognizer.load_state_dict(torch.load("digit_recognition.pt"))
        digit_recognizer.eval()

        img = Image.open("handwritten_digit.png")
        img_tensor = ToTensor()(img).unsqueeze(0).type(torch.float32).cuda()
        prediction = digit_recognizer(img_tensor).argmax().item()

        self.prediction_label.config(text="The predicted digit is: " + str(prediction))

        # Reset the image to a new blank image after prediction
        self.image = Image.new("L", (280, 280), "black")
        self.draw_tool = ImageDraw.Draw(self.image)


    def exit_program(self):
        self.master.destroy()
        sys.exit()

# Create CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout(p=0.4),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(p=0.4),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(64*28*28, 10)
        )

    def forward(self, x):
        return self.model(x)

def main():
    root = tk.Tk()
    app = HandwrittenDigitGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
