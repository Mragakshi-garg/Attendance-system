import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime

# Global cache so we don't re-encode all student images every time the camera opens!
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_NAMES = []
KNOWN_FACE_IDS = []
ALREADY_MARKED_TODAY = set()

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_app():
    """Fetches newly registered students from the DB and generates their face encodings in global memory."""
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES, KNOWN_FACE_IDS, ALREADY_MARKED_TODAY
    
    print("Checking for new faces in database...")
    conn = get_db_connection()
    students = conn.execute('SELECT * FROM students').fetchall()
    
    # If the number of students in DB matches our cache, we don't need to do anything
    if len(students) != len(KNOWN_FACE_NAMES):
        print("Reloading face encodings into Global RAM Cache...")
        KNOWN_FACE_ENCODINGS.clear()
        KNOWN_FACE_NAMES.clear()
        KNOWN_FACE_IDS.clear()
        
        for student in students:
            try:
                image = face_recognition.load_image_file(student['encoding_path'])
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    encoding = encodings[0]
                    KNOWN_FACE_ENCODINGS.append(encoding)
                    KNOWN_FACE_NAMES.append(student['name'])
                    KNOWN_FACE_IDS.append(student['id'])
                    print(f"Loaded encoding for: {student['name']}")
                else:
                    print(f"WARNING: No face found in image for {student['name']}")
            except Exception as e:
                print(f"Error loading image for {student['name']}: {e}")
    
    # Load today's attendance
    today = datetime.now().strftime("%Y-%m-%d")
    records = conn.execute('SELECT student_id FROM attendance WHERE date = ?', (today,)).fetchall()
    ALREADY_MARKED_TODAY.clear()
    for r in records:
        ALREADY_MARKED_TODAY.add(r['student_id'])
    
    conn.close()
    print(f"Total faces loaded in memory: {len(KNOWN_FACE_NAMES)}")

def mark_attendance(student_id):
    """Logs the student into the database if they haven't been seen today."""
    global ALREADY_MARKED_TODAY
    if student_id not in ALREADY_MARKED_TODAY:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        conn = get_db_connection()
        conn.execute('INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)',
                     (student_id, date_str, time_str, 'Present'))
        conn.commit()
        conn.close()
        
        ALREADY_MARKED_TODAY.add(student_id)
        print(f"[*] Attendance marked for Student ID {student_id} at {time_str}")

def recognize_frame(frame):
    """Receives a BGR numpy frame, performs face recognition, marks attendance, returns bounding boxes."""
    print("DEBUG: Frame shape:", frame.shape)
    
    # Convert BGR (OpenCV) to RGB (face_recognition) using cvtColor
    # This guarantees a C-contiguous uint8 array which dlib strictly requires!
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    print("DEBUG: Faces detected:", len(face_locations))
    
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    
    if face_encodings:
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            student_id = None
            
            if len(KNOWN_FACE_ENCODINGS) > 0:
                # Pass the native Python list (KNOWN_FACE_ENCODINGS) directly!
                # Wrapping it in np.array() causes dlib's PyBind11 module to crash.
                face_distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
                
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < 0.65:
                    name = KNOWN_FACE_NAMES[best_match_index]
                    student_id = KNOWN_FACE_IDS[best_match_index]
                    
                    mark_attendance(student_id)
            
            results.append({
                "name": name,
                "box": {"top": top, "right": right, "bottom": bottom, "left": left}
            })
            
    return results
