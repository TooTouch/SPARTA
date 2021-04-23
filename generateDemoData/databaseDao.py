import cx_Oracle

conn = cx_Oracle.connect(id,password,'163.152.183.95:9000/XE')

def insert(stmtSql, value):    
    cursor = conn.cursor()
    cursor.execute(stmtSql, value)    
    cursor.close()
def select(stmtSql):
    cursor = conn.cursor()
    cursor.execute(stmtSql)
    return cursor
def commit():
    conn.commit()
def close():
    conn.close()
