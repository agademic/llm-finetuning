import sqlite3

conn = sqlite3.connect('inputs.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS inputs
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              instruction TEXT,
              input TEXT,
              output TEXT)''')

conn.commit()
conn.close()
