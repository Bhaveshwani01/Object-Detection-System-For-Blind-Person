# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define a dictionary mapping object labels to corresponding voice messages
voice_messages = {
    "dog": "Dog detected",
    "cat": "Cat detected",
    "person": "Person detected",
    "bottle": "Bottle detected",
    "car": "Car detected",
    "background": "background detected",
    "aeroplane": "aeroplane detected",
    "bicycle": "bicycle detected",
    "bird": "bird detected",
    "boat": "boat detected",
    "bus": "bus detected",
    "cow": "Cat detected",
    "dining table": "dining table detected",
    "sofa": "sofa detected",
    "train": "train detected",

    ##mam
    "mouse": "Mouse detected",
    "laptop": "Laptop detected",
    "keyboard": "Keyboard detected",
    "chair": "Chair detected",
    "table": "Table detected",
    "door": "Door detected",
    "bucket": "Bucket detected",
    "pen": "Pen detected",
    "projector": "Projector detected",
    "vehicle": "Vehicle detected",
    "mobile": "Mobile detected",
    "steps": "Steps detected",
    "traffic light": "Traffic Light detected",
    "coin": "Coin detected",
    "rupees": "Rupees detected",
}

# Load the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tv monitor", "mouse", "laptop", "keyboard", "table",
           "door", "bucket", "pen", "projector", "vehicle", "mobile", "steps",
           "traffic light", "coin", "rupees"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained MobileNet SSD model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.2:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # Check if the detected label has a corresponding voice message
            if label.lower() in voice_messages:  # Convert to lowercase for case-insensitive matching
                # Speak the corresponding voice message for the detected object
                engine.say(voice_messages[label.lower()])  # Convert label to lowercase for dictionary lookup
                engine.runAndWait()

            # Draw the bounding box and label on the frame
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
