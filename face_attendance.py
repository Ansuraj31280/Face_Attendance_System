import cv2
import numpy as np
import os
import csv
import datetime

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Manually set the path to the Haar cascade file
cascade_path = r'C:\Users\ansur\OneDrive\Desktop\face_attendance_system\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load known faces and names from files (if available)
known_faces = []
known_names = []

if os.path.exists('known_faces.npy'):
    known_faces = list(np.load('known_faces.npy', allow_pickle=True))
if os.path.exists('known_names.npy'):
    known_names = list(np.load('known_names.npy', allow_pickle=True))

# Load registered users from the CSV file
registered_users = {}
if os.path.exists('registered.csv'):
    with open('registered.csv', mode='r') as file:
        reader = csv.reader(file)
        registered_users = {rows[0]: rows[1] for rows in reader}

def save_known_faces():
    np.save('known_faces.npy', np.array(known_faces, dtype=object))
    np.save('known_names.npy', np.array(known_names, dtype=object))

def register_face():
    global known_faces, known_names
    name = input("Enter your name: ")
    print("Please look at the camera and press 's' to capture your face.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image for registration")
            return
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_image = gray_frame[y:y+h, x:x+w]
                face_image_resized = cv2.resize(face_image, (100, 100))  # Resize to a fixed size
                known_faces.append(face_image_resized)
                known_names.append(name)
                save_known_faces()
                with open('registered.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, datetime.datetime.now()])
                print("Face registered successfully!")
            else:
                print("No face detected. Please try again.")
            break

def mark_attendance(name):
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.datetime.now()])

def recognize_face(face_image):
    face_image_resized = cv2.resize(face_image, (100, 100))  # Resize to match the registered face size
    for i, known_face in enumerate(known_faces):
        known_face = np.array(known_face, dtype=np.uint8)  # Ensure the known face is in the correct format
        result = cv2.matchTemplate(face_image_resized, known_face, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > 0.6:  # Adjust this threshold based on your testing
            return known_names[i]
    return "Unknown"

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_names = []
    for (x, y, w, h) in faces:
        face_image = gray_frame[y:y+h, x:x+w]
        name = recognize_face(face_image)
        face_names.append(name)

        # Draw a box around the face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('r'):
        register_face()
    elif key == ord('p'):
        for name in face_names:
            if name != "Unknown":
                mark_attendance(name)
                print(f"Attendance marked for {name}")
    elif key == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
