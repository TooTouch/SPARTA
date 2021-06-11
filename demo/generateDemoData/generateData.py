import databaseDao as dao
import datetime
import random
import string
from collections import defaultdict

#반송 완료 시점 기준점
CREATETIME = datetime.datetime.now()
#event prefix 컬럼
timeColumnsCount = 7
#생성할 Data
generateNum = 100000

carCount =50
devicesCount =500
vehicleCount = 500

carriers = [carCount]
devices = [devicesCount]
vehicles = [vehicleCount]
lineId = [1,2,3,4,5]

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def dev_generator(size=6, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))
def vhl_generator(size=3, chars= string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def error_generator(size=4, chars= string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generateTransferData(curTime, timeInterval, rowCount, resCount): 
    errorNum = 0   
    if rowCount <= 0:
        print("Wrong rowCount Number")
        return
    for i in range(1,rowCount+1):            
        insertSql = "INSERT INTO TRANSFER (CAR_ID, ROBOT_ID, FROMDEVICE, TODEVICE, CREATETIME, COMMAND_RECEIVED, COMMAND_START, UNLOAD_START, \
             UNLOAD_COM, LOAD_START, LOAD_COM, LINEID, SCHEDULED_QUEUE, UNLOADTIME, LOADTIME) \
                 VALUES (:1,:2,:3,:4,TO_DATE(:5,'YYYY/MM/DD hh24:mi:ss'),\
                     TO_DATE(:6,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:7,'YYYY/MM/DD hh24:mi:ss'),\
                         TO_DATE(:8,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:9,'YYYY/MM/DD hh24:mi:ss'),\
                             TO_DATE(:10,'YYYY/MM/DD hh24:mi:ss'),\
                                 TO_DATE(:11,'YYYY/MM/DD hh24:mi:ss'),:12,\
                                     ROUND((TO_DATE(:7,'YYYY/MM/DD hh24:mi:ss') - TO_DATE(:6,'YYYY/MM/DD hh24:mi:ss'))*1440*60),\
                                         ROUND((TO_DATE(:9,'YYYY/MM/DD hh24:mi:ss') - TO_DATE(:8,'YYYY/MM/DD hh24:mi:ss'))*1440*60),\
                                             ROUND((TO_DATE(:11,'YYYY/MM/DD hh24:mi:ss') - TO_DATE(:10,'YYYY/MM/DD hh24:mi:ss'))*1440*60))"        

        curTime = curTime -timeInterval
        timeList= [timeColumnsCount+1]    
        time = curTime
        
        carId = str(carriers[random.randrange(1,carCount-1)])
        #robotId = str(vehicles[random.randrange(1,vehicleCount-1)])
        robotList = list(vhlmap.keys())
        robotId = random.choice(robotList)
        fromDev = str(devices[random.randrange(1,devicesCount-1)])
        toDev= str(devices[random.randrange(1,devicesCount-1)])
        line = vhlmap[robotId]

        for j in range (1, timeColumnsCount+1):        
            varSeconds = random.randrange(0,30)        
            time = time - datetime.timedelta(seconds=varSeconds)
            timeList.append(time)
        
        createTime = str(timeList[1].strftime("%Y/%m/%d %H:%M:%S")) 
        cmdReceivedTime = str(timeList[6].strftime("%Y/%m/%d %H:%M:%S"))
        cmdStartTime = str(timeList[5].strftime("%Y/%m/%d %H:%M:%S"))
        unloadStartTime = str(timeList[4].strftime("%Y/%m/%d %H:%M:%S"))
        unloadCompTime = str(timeList[3].strftime("%Y/%m/%d %H:%M:%S"))
        loadStartTime =  str(timeList[2].strftime("%Y/%m/%d %H:%M:%S"))
        loadCompTime =  str(timeList[1].strftime("%Y/%m/%d %H:%M:%S"))
        
        schedQTime  = int(random.randrange(30))
        unloadTime = int(random.randrange(30))
        loadTime = int(random.randrange(30))        
          
        values = (carId, robotId, fromDev, toDev, createTime, cmdReceivedTime, cmdStartTime, unloadStartTime, unloadCompTime, loadStartTime, loadCompTime, line)

        dao.insert(insertSql, values)
        resCount +=1                
        if (resCount % 5000 == 0):
            print("Commit",resCount, "/",rowCount)
            print("Transfer History Inserted...",resCount, "/",rowCount)
            dao.commit()
        
        errorProb = random.randrange(0,1000)        
        if errorProb < 10:
            errordevice = random.choice([fromDev, toDev])
            errorNum +=1
            generateErrorData(carId, robotId, errordevice,unloadStartTime,unloadCompTime,loadStartTime,loadCompTime,errorNum)

    return resCount

def generateErrorData(carId, robotId, errordevice, unloadStartTime,unloadCompTime,loadStartTime,loadCompTime,errorNum):
    insertSql = "INSERT INTO ERROR_HIST (CAR_ID, ROBOT_ID, DEVICE, DEVICEPORT, ERRORCODE, ERROR_SET, ERROR_CLEAR, RECOVERYTIME) \
        VALUES (:1,:2,:3,:4,:5,\
            TO_DATE(:6,'YYYY/MM/DD hh24:mi:ss'),\
                TO_DATE(:7,'YYYY/MM/DD hh24:mi:ss'),\
                    ROUND((TO_DATE(:7,'YYYY/MM/DD hh24:mi:ss')-TO_DATE(:6,'YYYY/MM/DD hh24:mi:ss'))*1440*60))"
    errorPort = random.choice(['B1','B2','B3','B4'])
    errorCode = error_generator()
    errorType = random.choice(['UNLOAD','LOAD'])
    if (errorType == 'UNLOAD'):
        error_set = unloadStartTime
        error_clear = unloadCompTime
    else:
        error_set = loadStartTime
        error_clear = loadCompTime
    
    values = (carId, robotId, errordevice, errorPort, errorCode,error_set,error_clear)
    dao.insert(insertSql, values)    
    print("Error History Inserted...(",errorNum,")")
    dao.commit()   
    
vhlmap = defaultdict(str)
        
for i in range(carCount):
        carriers.append(id_generator())        
for i in range(devicesCount):     
        devices.append(dev_generator())
for i in range(vehicleCount):     
        #vehicles.append(vhl_generator())
        vhlId = vhl_generator()
        if vhlId in vhlmap:
            continue
        vhlmap[vhlId] = lineId[random.randrange(0,5)]

result = 0
curTime = CREATETIME
transferTimeInterval = datetime.timedelta(seconds=5)
print("Transfer inserted rows : ", generateTransferData(curTime, transferTimeInterval, generateNum, result))
dao.close()