import os
import cv2
import base64
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import camera
import pandas as pd

app = Flask(__name__)
app.secret_key = 'super_secret_key_attendance'

# Load face encodings into memory once on startup
try:
    with app.app_context():
        camera.init_app()
except Exception as e:
    print("Warning: Could not initialize camera DB on startup.")

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        department = request.form.get('department')
        image_data = request.form.get('image_data')  # Get Base64 image from frontend JS
        
        if not name or not department:
            flash("Name and Department are required.")
            return redirect(url_for('register'))
            
        if not image_data:
            flash("Could not detect webcam image. Please try again.")
            return redirect(url_for('register'))
            
        print(f"Registering student: {name} ({department})")
        
        # 1. Process the Base64 Image
        # The js string looks like "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
        # We need to strip out the header to decode it
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # 2. Convert base64 bytes to numpy array, then to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 3. Save Image
        faces_dir = os.path.join('static', 'faces')
        os.makedirs(faces_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_name = name.replace(" ", "_").replace("/", "")
        filename = f"{safe_name}_{timestamp}.jpg"
        filepath = os.path.join(faces_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Saved face image to {filepath}")
        
        # 4. Save to Database
        conn = get_db_connection()
        conn.execute('INSERT INTO students (name, department, encoding_path) VALUES (?, ?, ?)',
                     (name, department, filepath))
        conn.commit()
        conn.close()
        
        # Manually reload the RAM cache so the new student is instantly recognized
        camera.init_app()
        
        flash(f"Student {name} registered successfully!")
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/manage')
def manage():
    conn = get_db_connection()
    students = conn.execute('SELECT * FROM students ORDER BY id DESC').fetchall()
    conn.close()
    return render_template('manage.html', students=students)

@app.route('/delete_student/<int:id>', methods=['POST'])
def delete_student(id):
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE id = ?', (id,)).fetchone()
    
    if student:
        # Delete image file
        filepath = student['encoding_path']
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Deleted face image: {filepath}")
            except Exception as e:
                print(f"Error deleting file {filepath}: {e}")
                
        # Delete from DB
        conn.execute('DELETE FROM students WHERE id = ?', (id,))
        # Also delete their attendance records to maintain relational integrity
        conn.execute('DELETE FROM attendance WHERE student_id = ?', (id,))
        conn.commit()
        
        # Reload cache so the deleted student is removed from RAM
        camera.init_app()
        flash(f"Student {student['name']} deleted successfully!")
    else:
        flash("Student not found.")
        
    conn.close()
    return redirect(url_for('manage'))

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """Receives a base64 image from the frontend, detects faces, and logs attendance."""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400
        
    image_data = data['image']
    try:
        # Strip header
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # DEBUG: Save exactly what the server received so we can see what's wrong
        cv2.imwrite("debug_capture.jpg", frame)
        print(f"DEBUG: Saved incoming API frame of size {frame.shape}")
        
        # Analyze the frame instantly using the global cache
        results = camera.recognize_frame(frame)
        return jsonify({"results": results})
        
    except Exception as e:
        print(f"Error during recognition API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard')
def dashboard():
    conn = get_db_connection()
    today = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    
    # 1. Get total distinct students in each department
    departments_query = conn.execute('''
        SELECT department, COUNT(id) as total_students 
        FROM students 
        GROUP BY department
    ''').fetchall()
    
    # 2. Get today's attendance grouped by department
    attendance_query = conn.execute('''
        SELECT s.department, COUNT(DISTINCT a.student_id) as present_count
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        GROUP BY s.department
    ''', (today,)).fetchall()
    
    # 3. Create a clean dictionary mapping: Department -> {total, present, absent}
    stats = {}
    for row in departments_query:
        stats[row['department']] = {'total': row['total_students'], 'present': 0}
        
    for row in attendance_query:
        if row['department'] in stats:
            stats[row['department']]['present'] = row['present_count']
            
    for dept in stats:
        stats[dept]['absent'] = stats[dept]['total'] - stats[dept]['present']
        
    # 4. Fetch the raw logs for a table at the bottom of dashboard
    logs = conn.execute('''
        SELECT s.name, s.department, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    ''', (today,)).fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', stats=stats, logs=logs, today=today)

@app.route('/api/recent_attendance')
def recent_attendance():
    """Returns the 5 most recently recognized students for the dynamic UI."""
    conn = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")
    logs = conn.execute('''
        SELECT s.name, s.department, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.time DESC
        LIMIT 5
    ''', (today,)).fetchall()
    conn.close()
    
    return {"recent": [dict(row) for row in logs]}

@app.route('/export_attendance')
def export_attendance():
    conn = get_db_connection()
    today = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    
    query = '''
        SELECT a.date, a.time, s.name, s.department, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    '''
    df = pd.read_sql_query(query, conn, params=(today,))
    conn.close()
    
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance_report.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
