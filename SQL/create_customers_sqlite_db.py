#%% Import Libraries

import sqlite3

#%% Connect to SQLite

conn = sqlite3.connect("customers.db")
cursor = conn.cursor()

#%% Create Table

cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT NOT NULL,
        brand TEXT NOT NULL,
        series TEXT NOT NULL,
        price INTEGER NOT NULL)
        ''')

#%% Insert DATA

customers_data = [
    ("John", "BMW", "M4", 4500000),
    ("Steve", "Mercedes", "AMG C63", 4750000),
    ("Luke", "Audi", "RS4", 4330000),
    ("Tom", "BMW", "M5", 6550000),
    ("Amber", "Mercedes", "AMG E63", 6880000),
    ("Cynthia", "Audi", "RS7", 6480000)]

cursor.executemany("INSERT INTO customers (customer_name, brand, series, price) VALUES (?, ?, ?, ?)", customers_data)

#%% Display Content of STUDENTSs.db

print('The Content of customers.db')
data = cursor.execute('select * from customers')
for row in data:
    print(row)

#%% Commit the Changes

conn.commit()
conn.close()
