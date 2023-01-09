from CLUSTER import cluster
import os
import pickle
import time
import joblib

def get_time():
    st = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print(st)
    return st

def prepareSaveDir():
    # according to the time, get the save dir for model
    time = get_time()
    savedir = "..\\log\\Data\\" + time
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    f=open("..\\log\\logFile","a+")
    f.write("-------------------------------------\n")
    f.write("running time is: {}\n".format(time))
    f.close()
    return savedir

def read_dataset(filename): # read the dataset of CX and return x_train_origin, y_train
    #input: filename
    #output: clusterList, which is a list of cluster,class defined in CLUSTER.py describe the cluster
    if os.path.exists("../data/clusterLitByCXDataset"):
        print("---------------------there is already clusterList exit ----------------------------")
        cache = open("../data/clusterLitByCXDataset","rb")
        clusterList = pickle.load(cache)
        print("------------there are {} clusters ---------------".format(len(clusterList)))
        return clusterList    
    f=open(filename,'r')
    lines = f.readlines()
    clusterList=[]
    tick = 0
    for line in lines:
        information = line.split()
        information = [float(x) for x in information]
        newCluster = cluster(information)
        clusterList.append(newCluster)
        if tick % 10000 == 0:
                print("I have finish {} reading in function read_dataset".format(tick))  
        tick+=1
    print("------------------------I am happy, I have finished the work--------------------------------") 
    print("-----------------now cache the reslut in clusterLitByCXDataset file-------------------------")    
    cache = open("../data/clusterLitByCXDataset","wb")
    pickle.dump(clusterList,cache,0)
    f.close
    return clusterList

def cacheData(dataStructure,filename):

    f = open(filename,"wb")
    pickle.dump(dataStructure,f,0)
    f.close()

def loadCache(filename):
    cache = open(filename,"rb")
    target = pickle.load(cache)
    return target

def writeLog(string:str, logName): # write string to logFile, each time will start a new line
    f = open(logName,"a+")
    f.write(string+"\n")
    f.close()

def cacheByJobLib(dataStructure,filename):
    joblib.dump(dataStructure,filename)

def loadJoblib(filename):
    result=joblib.load(filename)
    return result

def makeposcar(target_pos,lattice,pos):# write file poscar
    f = open(target_pos,'w')
    f.write('by code\n1.0\n')
    for item in lattice:
        f.write('%15.8f %15.8f %15.8f\n' % tuple(item))
    f.write('Li \n%d\ndirect\n'%len(pos))
    for item in pos:
        f.write('%15.8f %15.8f %15.8f\n' % tuple(item))
    f.close()

def point2poscar(target_pos,point):
    LATIICE_n=round(len(point)/2)+15
    lattice_tmp=[[LATIICE_n,0,0],[0,LATIICE_n,0],[0,0,LATIICE_n]]
    tem_point=[]
    for item in point:
        tem_point.append(list(map(lambda x:x/LATIICE_n+0.5,item)))
    makeposcar(target_pos,lattice_tmp,tem_point)