import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained age prediction model
try:
    model = load_model('C:/Users/Uday Alugolu/OneDrive/Desktop/AgePredictionProject/model/age_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the face image for age prediction
def preprocess_image(face):
    # Resize the face image to 64x64 for the model input
    face_resized = cv2.resize(face, (64, 64))
    face_resized = face_resized.astype('float32') / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension
    return face_resized

# Initialize the camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for age prediction
        preprocessed_face = preprocess_image(face)
        
        # Predict the age using the model
        try:
            age_prediction = model.predict(preprocessed_face)
            predicted_age = int(age_prediction[0][0])  # The model outputs the estimated age
        except Exception as e:
            print(f"Error during prediction: {e}")
            predicted_age = "N/A"

        # Draw a rectangle around the face and display the predicted age
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{predicted_age} years old', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Age Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
