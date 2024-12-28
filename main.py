import numpy as np
import cv2
import os
import math
import PIL
from PIL import ImageTk
import PIL.Image
import speech_recognition as sr
import pyttsx3
from tkinter import *
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

try:
    import tkinter as tk
except ImportError:
    import tkinter as tk

# Paths
filtered_data_folder = r"C:\Users\Acer\Downloads\project3\filtered_data"
alphabet_folder = r"C:\Users\Acer\Downloads\project3\images"
model_path = r"C:\Users\Acer\Downloads\project3\keras_model.h5"
labels_path = r"C:\Users\Acer\Downloads\project3\labels.txt"

# Initialize Hand Detector and Classifier
detector = HandDetector(maxHands=2)
classifier = Classifier(model_path, labels_path)

# Constants
offset = 20
imgSize = 300
labels = [chr(i) for i in range(65, 91)]  # A-Z labelsf

# Store current detected sign temporarily
current_sign = ""

# Generate frames for input text
def func(input_text):
    all_frames = []
    words = input_text.split()

    for word in words:
        word_path = os.path.join(filtered_data_folder, f"{word}.webp")

        if os.path.exists(word_path):
            print(f"Found: {word}")
            im = PIL.Image.open(word_path)

            # Load all frames from the .webp file
            for frame_cnt in range(im.n_frames):
                im.seek(frame_cnt)
                duration = im.info.get('duration', 50)  # Default duration is 100ms if not specified
                img = im.convert("RGB").resize((800, 500))  # Resize for better visibility
                all_frames.append((img, duration))
        else:
            print(f"Not Found: {word}, breaking into characters")
            for char in word:
                char_path = os.path.join(alphabet_folder, f"{char.lower()}.jpg")

                if os.path.exists(char_path):
                    im = PIL.Image.open(char_path)
                    img = im.resize((800, 500))  # Resize for better visibility
                    all_frames.append((img, 1000))  # Display static JPG images for 1 second
                else:
                    print(f"Character image not found for: {char}")

    return all_frames

# Play frames dynamically in the GIF box
def gif_stream(frames, gif_box):
    for img, duration in frames:
        imgtk = ImageTk.PhotoImage(image=img)
        gif_box.imgtk = imgtk
        gif_box.configure(image=imgtk)
        gif_box.update()
        gif_box.after(duration)  # Use the frame duration

# Predict gesture from webcam feed using HandDetector and Classifier
def process_hand_sign(video_label, text_area):
    cap = cv2.VideoCapture(0)

    def update_frame():
        global current_sign
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            return

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure coordinates are within image boundaries
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                aspectRatio = h / w

                if aspectRatio > 1:  # Height > Width
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:  # Width > Height
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Predict gesture and get confidence
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                confidence = prediction[index]

                # Draw a black rectangle for the label
                cv2.rectangle(imgOutput, (x1, y1 - 60), (x2, y1), (0, 0, 0), cv2.FILLED)
                text = f"{labels[index]} ({confidence * 100:.2f}%)"
                cv2.putText(imgOutput, text, (x1 + 10, y1 - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

                # Update the current_sign with detected label
                current_sign = labels[index]

                # Draw rectangle around detected hand
                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Convert frame to RGB and display in video label
        rgb_frame = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, update_frame)

    update_frame()

# Convert text to speech
def text_to_speech(text_area):
    text = text_area.get("1.0", "end-1c").strip()
    if text:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        print("Text area is empty.")

# GUI Class
class Tk_Manage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, VtoS, StoV):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

# Start Page
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Two Way Sign Language Translator", font=("Verdana", 12))
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="VOICE TO SIGN", command=lambda: controller.show_frame(VtoS))
        button1.pack(pady=5)
        button2 = tk.Button(self, text="SIGN TO VOICE", command=lambda: controller.show_frame(StoV))
        button2.pack(pady=5)

        img = PIL.Image.open(r"C:\Users\Acer\Downloads\project3\Two Way Sign Language Translator.png").resize((800, 550))
        render = ImageTk.PhotoImage(img)
        img_label = Label(self, image=render)
        img_label.image = render
        img_label.pack(pady=10)
# Speech recognition to record voice and convert to text
def record_voice_to_text(input_txt_widget):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Recording... Speak now.")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        input_txt_widget.insert(tk.END, text)
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results from the speech recognition service.")

# Voice to Sign (Text to Sign)
class VtoS(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Voice to Sign", font=("Verdana", 12))
        label.pack(pady=10, padx=10)

        split_frame = tk.Frame(self)
        split_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(split_frame, bg="black")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame = tk.Frame(split_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame.columnconfigure(0, weight=1)

        self.inputtxt = tk.Text(right_frame, height=15, width=60, font=("Verdana", 14))
        self.inputtxt.grid(row=0, column=0, pady=20)

        # Create a horizontal frame for buttons (like in StoV)
        buttons_frame = tk.Frame(right_frame)
        buttons_frame.grid(row=1, column=0, pady=20)

        # Buttons arranged horizontally
        tk.Button(buttons_frame, text="Record", command=lambda: record_voice_to_text(self.inputtxt),
                  font=("Verdana", 14), height=2, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Convert", command=lambda: self.process_text(gif_box),
                  font=("Verdana", 14), height=2, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Back to Home", command=lambda: controller.show_frame(StartPage),
                  font=("Verdana", 14), height=2, width=15).pack(side=tk.LEFT, padx=5)

        gif_box = tk.Label(left_frame, bg="black", text="Images/GIFs will appear here", fg="white", font=("Verdana", 10))
        gif_box.pack(fill=tk.BOTH, expand=True)

    def process_text(self, gif_box):
        text = self.inputtxt.get("1.0", "end-1c").strip()

        if text:
            frames = func(text)
            gif_stream(frames, gif_box)


# Sign to Voice
class StoV(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Sign to Voice", font=("Verdana", 12))
        label.pack(pady=10, padx=10)

        split_frame = tk.Frame(self)
        split_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(split_frame, bg="black")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=25, pady=30)

        right_frame = tk.Frame(split_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_label = tk.Label(left_frame, bg="black")
        video_label.pack(fill=tk.BOTH, expand=True)

        text_area = tk.Text(right_frame, font=("Helvetica", 14), wrap=tk.WORD, height=15, width=35)
        text_area.pack(pady=10)

        buttons_frame = tk.Frame(right_frame)
        buttons_frame.pack(pady=10)

        tk.Button(buttons_frame, text="Add Sign", font=("Verdana", 12),
                  command=lambda: text_area.insert(tk.END, current_sign)).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Clear All", font=("Verdana", 12),
                  command=lambda: text_area.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Back to Home", font=("Verdana", 12),
                  command=lambda: controller.show_frame(StartPage)).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Quit", font=("Verdana", 12), bg="red",
                  command=controller.quit).pack(side=tk.LEFT, padx=5)

        tk.Button(self, text="Voice", font=("Verdana", 14), bg="grey", fg="white",
                  command=lambda: text_to_speech(text_area)).pack(pady=20)

        process_hand_sign(video_label, text_area)

# Run the App
app = Tk_Manage()
app.geometry("800x750")
app.mainloop()

