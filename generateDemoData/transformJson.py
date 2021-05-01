import databaseDao as dao
import datetime
import random
import string
import cx_Oracle

tableIdx = "1-100000-10"
table = "BANK"

def getColumn(tableName):
    selectSql = "SELECT COLUMN_NAME FROM all_tab_columns WHERE 1=1 AND TABLE_NAME ='"+tableName+"' ORDER BY COLUMN_ID"    
    cursor = dao.select(selectSql)
    columns ='"header": '
    rowCount =0
    for row in cursor:        
        for i in range(len(row)):
            if rowCount == 0:
                columns = columns+'["'+row[i]
            else:
                columns = columns+'", "'+row[i]     
        rowCount +=1
    columns = columns +'"]'
    print(columns)    
    return columns

def getRows(tableName, cond):
    if cond is None:
        cond = " 1=1 "
    selectSql = "SELECT * FROM "+tableName+" WHERE 1=1" + " AND " +cond
    cursor = dao.select(selectSql)
    
    resultStr = '"rows":['
    rowStr =''
    rowCount =0    
    for row in cursor:        
        for i in range(len(row)):            
            if i == 0:
                print(row[i])
                if (isinstance(row[i], datetime.date)):                                                            
                    rowStr = '["'+row[i].strftime("%Y/%m/%d %H:%M:%S")
                elif (isinstance(row[i], int) or isinstance(row[i], float) ):
                    rowStr = '["'+str(row[i])
                else:
                    rowStr = '["'+row[i]
            else:
                if (isinstance(row[i], datetime.date)):                                                            
                    rowStr = rowStr+', '+ '"' +row[i].strftime("%Y/%m/%d %H:%M:%S") +'"'
                elif (isinstance(row[i], int) or isinstance(row[i], float)):
                    rowStr = rowStr+', '+str(row[i])
                else:
                    rowStr = rowStr+', '+'"'+row[i]+'"'
        rowStr = rowStr +']'                
        if rowCount == 0:
            resultStr = resultStr+rowStr
        else:
            resultStr = resultStr +', ' +rowStr
        rowStr=""
        rowCount +=1
    resultStr = resultStr +']'
    print(resultStr)    
    return resultStr

def getTypes(tableName):
    selectSql = "SELECT DATA_TYPE FROM all_tab_columns WHERE 1=1 AND TABLE_NAME ='"+tableName+"' ORDER BY COLUMN_ID"
    cursor = dao.select(selectSql)
    typeStr ='"types": '
    rowCount =0
    for row in cursor:        
        for i in range(len(row)):
            coltypes ="text"
            if row[i] == "NUMBER":
                coltypes = "real"            
            if rowCount == 0:
                typeStr = typeStr+'["'+coltypes
            else:
                typeStr = typeStr+'", "'+coltypes     
        rowCount +=1
    typeStr = typeStr +'"]'
    print(typeStr)    
    return typeStr

def createJson(tableName):
    f = open("test2.tables.jsonl",'w')
    tableStr = '{"id": "'+tableIdx+'", ' + getColumn(tableName) +", " +getTypes(tableName)+", " +getRows(tableName, None) +"}"
    f.write(tableStr)
    f.close()    

createJson(table)


