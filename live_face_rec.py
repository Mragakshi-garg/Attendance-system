import cv2
import face_recognition

# Start webcam
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
try:
    print("Trying to load Mragakshi.jpg...")
    # Make sure you have a picture named "Mragakshi.jpg" in the Attendance folder
    my_image = face_recognition.load_image_file("Mragakshi.jpg")
    my_face_encoding = face_recognition.face_encodings(my_image)[0]
    known_face_encodings = [my_face_encoding]
    known_face_names = ["Mragakshi"]
    print("Successfully loaded face encoding!")
except Exception as e:
    print(f"Could not load image: {e}. Running without recognized faces.")
    known_face_encodings = []
    known_face_names = []

print("Starting live webcam feed. Press 'q' to quit.")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    if not ret or frame is None:
        continue

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Find all faces & encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = []
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Display the result
    cv2.imshow('Face Recognition Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
