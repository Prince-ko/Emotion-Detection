import cv2
import numpy as np
from keras.models import model_from_json
import os

# Define emotion labels
emotion_labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Model paths
model_structure_path = 'model/emotion_model.json'
model_weights_path = 'model/emotion_model.weights.h5'

# Load the model structure and weights
if not os.path.exists(model_structure_path) or not os.path.exists(model_weights_path):
    print("Model or weights file not found.")
    exit()

with open(model_structure_path, 'r') as json_file:
    model_json = json_file.read()

emotion_model = model_from_json(model_json)
emotion_model.load_weights(model_weights_path)
print("Model loaded successfully.")

# Haar cascade path
cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print("Haar cascade file missing.")
    exit()

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load video
# video_file = r"F:\Project\Emotion_detection_with_CNN-main\Sample_videos\WIN_20240921_10_17_36_Pro.mp4"
video_file = 0
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Failed to open video.")
    exit()

# Loop through video frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or error reading frame.")
        break
    frame = cv2.resize(frame, (640, 360))
    # Adjust brightness and contrast
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=40)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_area = gray[y:y + h, x:x + w]
        resized_face = np.expand_dims(np.expand_dims(cv2.resize(face_area, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_probabilities = emotion_model.predict(resized_face)
        emotion_index = int(np.argmax(emotion_probabilities))
        emotion_label = emotion_labels[emotion_index]

        # Display results
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 3)
        cv2.putText(frame, emotion_label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Quit video on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
