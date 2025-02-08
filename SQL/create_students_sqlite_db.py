#%% Import Libraries

import sqlite3

#%% Connect to SQLite

connection = sqlite3.connect('students.db')

#%% Create Cursor

cursor = connection.cursor()

#%% Create Table

table_info = """
    create table STUDENTS(NAME VARCHAR(25), CLASS VARCHAR(25), 
    SECTION VARCHAR(25), SCORE INT)
    """

cursor.execute(table_info)

#%% Insert DATA

cursor.execute('''Insert Into STUDENTS values('Krish','Data Science','A',90)''')
cursor.execute('''Insert Into STUDENTS values('John','Data Science','B',100)''')
cursor.execute('''Insert Into STUDENTS values('Mukesh','Data Science','A',86)''')
cursor.execute('''Insert Into STUDENTS values('Jacob','DEVOPS','A',50)''')
cursor.execute('''Insert Into STUDENTS values('Dipesh','DEVOPS','A',35)''')

#%% Display Content of STUDENTSs.db

print('The Content of stidents.db')
data = cursor.execute('select * from students')
for row in data:
    print(row)

#%% Commit the Changes

connection.commit()
connection.close()
