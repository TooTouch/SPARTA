import databaseDao as dao
import datetime
import random
import string

#반송 완료 시점 기준점
CREATETIME = datetime.datetime.now()
timeColumnsCount = 7
generateNum = 10000

carCount =50
devicesCount =50
vehicleCount = 50

carriers = [carCount]
devices = [devicesCount]
vehicles = [vehicleCount]

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def dev_generator(size=6, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))
def vhl_generator(size=2, chars= string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generateTransferData(curTime, timeInterval, rowCount, resCount):    
    if rowCount <= 0:
        print("Wrong rowCount Number")
        return
    for i in range(1,rowCount+1):            
        insertSql = "INSERT INTO TRANSFERHISTORY (CAR_ID, ROBOT_ID, FROMDEVICE, TODEVICE, CREATETIME, COMMAND_RECEIVED, COMMAND_START, UNLOAD_START, UNLOAD_COM, LOAD_START, LOAD_COM, SCHEDULED_QUEUE, UNLOADTIME, LOADTIME) VALUES (:1,:2,:3,:4,TO_DATE(:5,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:6,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:7,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:8,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:9,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:10,'YYYY/MM/DD hh24:mi:ss'),TO_DATE(:11,'YYYY/MM/DD hh24:mi:ss'),:12,:13,:14)"

        curTime = curTime -timeInterval
        timeList= [timeColumnsCount+1]    
        time = curTime
        
        carId = str(carriers[random.randrange(1,carCount-1)])
        robotId = str(vehicles[random.randrange(1,vehicleCount-1)])
        fromDev = str(devices[random.randrange(1,devicesCount-1)])
        toDev= str(devices[random.randrange(1,devicesCount-1)])

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
        
        values = (carId, robotId, fromDev, toDev, createTime, cmdReceivedTime, cmdStartTime, unloadStartTime, unloadCompTime, loadStartTime, loadCompTime, schedQTime, unloadTime, loadTime)

        dao.insert(insertSql, values)
        resCount +=1        
        print("Transfer History Inserted...",resCount, "/",rowCount)
        if (resCount % 5000 == 0):
            print("Commit",resCount, "/",rowCount)
            dao.commit()
        
    return resCount
        

        
for i in range(carCount):
        carriers.append(id_generator())        
for i in range(devicesCount):     
        devices.append(dev_generator())
for i in range(vehicleCount):     
        vehicles.append(vhl_generator())

result = 0
curTime = CREATETIME
timeInterval = datetime.timedelta(seconds=5)
print("Total inserted rows : ", generateTransferData(curTime, timeInterval, generateNum, result))
dao.close()