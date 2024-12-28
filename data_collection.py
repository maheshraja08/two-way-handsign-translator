import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Constants
offset = 20
imgSize = 300
counter = 0
image_limit = 60  # Maximum number of images to collect
folder = r"C:\Users\Acer\Downloads\project3\data\A"  # Folder path to save images

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Exiting...")
        break

    hands, img = detector.findHands(img)  # Detect hands

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region with an offset
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        # Check aspect ratio and resize accordingly
        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1:  # Height is greater than width
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:  # Width is greater than height
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display the cropped and white images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    # Display the main camera feed
    cv2.imshow('Image', img)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord("s") and counter < image_limit:  # Save image when 's' is pressed
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image {counter} saved.")
    elif key == ord("q"):  # Exit when 'q' is pressed
        print("Exiting...")
        break
    elif counter >= image_limit:  # Stop collecting images when the limit is reached
        print(f"Image collection limit of {image_limit} reached.")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
