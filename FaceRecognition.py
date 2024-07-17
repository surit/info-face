import os
import time
import cv2
import numpy as np

print("Initializing components...")

root = "data"
image_size = (200, 200)  # Common size for all images

# Initialize training data set and label set
training_data = []
training_labels = []
user_labels = []
label_counter = 0

# Initialize the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to capture and recognize faces
def capture_and_recognize():
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    start_time = time.time()
    end_time = start_time + 5.0  # Capture faces for 5 seconds

    faces_detected = False
    new_face_detected = False
    welcome_displayed = False

    while time.time() < end_time:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi = gray[y:y+h, x:x+w]

            # Resize the face region to the common size
            roi_resized = cv2.resize(roi, image_size)

            recognize_result, confidence = recognizer.predict(roi_resized)

            if recognize_result >= 0 and confidence < 100:
                cv2.putText(img, f"{user_labels[recognize_result - 1]} - {confidence:.2f}%", (x, y), font, 0.8, (255, 255, 255), 1)
                if confidence < 55 and not welcome_displayed:
                    print(f"Welcome {user_labels[recognize_result - 1]}")
                    faces_detected = True
                    welcome_displayed = True
            else:
                new_face_detected = True

        cv2.imshow('Face Recognition', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not faces_detected and new_face_detected:
        print("New face detected")

    return faces_detected

# Iterate over images in the data folder
for subdir, _, files in os.walk(root):
    for f in files:
        # Label of the person
        name = os.path.basename(subdir)

        # Path to the image file
        file_path = os.path.join(subdir, f)

        print(f"Processing image for {name}")

        # Read the image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            roi = img[y:y+h, x:x+w]

            # Resize the face region to the common size
            roi_resized = cv2.resize(roi, image_size)

            training_data.append(roi_resized)

            if name not in user_labels:
                user_labels.append(name)
                label_counter += 1

            training_labels.append(label_counter)
        else:
            print(f"Warning - File is not good: {file_path}")

print("Training data set...")
recognizer.train(np.asarray(training_data), np.asarray(training_labels))
print("Training complete!")

while True:
    print("Starting face detection...")
    if capture_and_recognize():
        print("Waiting for next scan...")
        time.sleep(10)  # Wait 10 seconds before next scan
    else:
        print("No face detected during scan.")
        time.sleep(10)  # Wait 10 seconds before next scan
