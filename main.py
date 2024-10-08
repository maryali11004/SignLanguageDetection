import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
from training import extract_keypoints, mediapipe_detection, draw_landmarks  # Importing from the training file
from tensorflow.keras.models import load_model  # Importing to load the trained model

# Constants
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
threshold = 0.5

# Load the trained model
model = load_model('action.h5')  # Ensure that the model file exists in the same directory

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()

# Initialize variables for predictions
sequence = []
sentence = []
predictions = []

# Function to visualize probabilities
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame

cap = cv2.VideoCapture(0)

# Set MediaPipe model 
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep the last 30 frames

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            # Visualization logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            text_to_speak = f'{actions[np.argmax(res)]}'
                            engine.say(text_to_speak)
                            engine.runAndWait()
                    else:
                        sentence.append(actions[np.argmax(res)])
                        text_to_speak = f'{actions[np.argmax(res)]}'
                        engine.say(text_to_speak)
                        engine.runAndWait()

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Draw visualization probabilities
            image = prob_viz(res, actions, image, colors)  # Use prob_viz to visualize probabilities

            # Display predicted gesture and accuracy
            text = f'Gesture: {actions[np.argmax(res)]} | Accuracy: {res[np.argmax(res)] * 100:.2f}%'
            cv2.putText(image, text, (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on ESC key
            break 
    
    cap.release()
    cv2.destroyAllWindows()

    for i in range(1, 5):
        cv2.waitKey(1)
