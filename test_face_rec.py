import cv2
import face_recognition
import sqlite3
import numpy as np
import pprint

def run_test():
    print("--- DEBUG FACE MATCHING VER 2 ---")
    
    # 1. Load the database encodings
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    students = conn.execute('SELECT * FROM students').fetchall()
    conn.close()
    
    known_encodings = []
    known_names = []
    
    print(f"Loading {len(students)} students from DB...")
    for s in students:
        try:
            img = face_recognition.load_image_file(s['encoding_path'])
            enc = face_recognition.face_encodings(img)
            if len(enc) > 0 and isinstance(enc[0], np.ndarray):
                print(f"  [+] Loaded {s['name']}: type={type(enc[0])} shape={enc[0].shape}")
                known_encodings.append(enc[0])
                known_names.append(s['name'])
            else:
                print(f"  [-] No face in {s['name']}")
        except Exception as e:
            print(f"  [X] Failed {s['name']}: {e}")
            
    print(f"Successfully loaded {len(known_encodings)} face encodings.")
    
    # 2. Analyze the debug capture
    print("Analyzing debug_capture.jpg...")
    frame = cv2.imread('debug_capture.jpg')
    rgb_frame = frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    print(f"Found {len(face_locations)} faces in the capture image.")
    
    if not face_encodings:
        print("No faces found in capture image to compare.")
        return
        
    capture_encoding = face_encodings[0]
    
    # 3. Match manually and print
    print("\n--- DISTANCES ---")
    
    # DO NOT wrap known_encodings in np.array(). PyBind11 exclusively expects a 
    # native python list containing individual numpy.ndarray objects!
    distances = face_recognition.face_distance(known_encodings, capture_encoding)
    
    for idx, dist in enumerate(distances):
        print(f"Distance to {known_names[idx]}: {dist}")
        
    best_match_index = np.argmin(distances)
    best_dist = distances[best_match_index]
    
    print(f"\nBest match is {known_names[best_match_index]} with distance {best_dist}")
    
    if best_dist < 0.6:
        print("MATCH WAS SUCCESSFUL (dist < 0.6)!")
    else:
        print("MATCH FAILED (dist >= 0.6). The capture is too different from the DB.")
        
    print("\nCapture array sum:", np.sum(capture_encoding))
    print("Best DB array sum:", np.sum(known_encodings[best_match_index]))

if __name__ == "__main__":
    run_test()
