import cx_Oracle

conn = cx_Oracle.connect('TEST','TEST','163.152.183.95:9000/XE')

def insert(stmtSql, value):    
    cursor = conn.cursor()
    cursor.execute(stmtSql, value)    
    cursor.close()
def commit():
    conn.commit()
def close():
    conn.close()
