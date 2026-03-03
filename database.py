import sqlite3
import os

DB_FILE = 'database.db'

def init_db():
    print("Initializing database...")
    # Connect to SQLite (this creates the file if it doesn't exist)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            encoding_path TEXT NOT NULL
        )
    ''')

    # Create attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized successfully at {os.path.abspath(DB_FILE)}.")

if __name__ == '__main__':
    init_db()
