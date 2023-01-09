import os
import sys
import algebra_graph
import io_routine
import numpy as np
import math
from sklearn import model_selection
from machineLearning import *
from sklearn import metrics
import pickle
from io_routine import cacheData, writeLog
import datetime
from multiprocessing import Pool
import argparse
import random
import fcntl

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.chdir(os.path.split(os.path.realpath(__file__))[0])
def main(args):
    Lin_train = io_routine.loadCache("../pretreat/Lin_train_n")
    Lin_val = io_routine.loadCache("../pretreat/Lin_val_n")
    Lin_test = io_routine.loadCache("../pretreat/Lin_test_n")
    Li20 = io_routine.loadCache("../pretreat/Li20_n")
    Li40 = io_routine.loadCache("../pretreat/Li40_n")


    #Exponential
    #Lorentz
    #需要写入log的量，etaE kappa, etaL,v； 每开始一个模型写一次，一个组参数重复r_times词
    #需要保存的文件：训练集，测试集，分别保存X Y， 保存训练模型——每个模型保存一次
    #etaE = list(map(lambda x:x/10, range(5,65,5)))
    #eta from 0.5 --- 6
    #kappa and v 0.5---6 10 15 20 

    '''soft参数
        n_estimators=15000
        subsample=0.7 
        min_samples_split=5
        max_depth=7 
    '''
    i=args.eta
    j=args.kappa
    l=args.v

    A_matByE=args.A_matByE
    L_matByE=args.L_matByE
    A_matByL=args.A_matByL
    L_matByL=args.L_matByL

    n_estimators=args.n_estimators
    subsample=args.subsample
    min_samples_split=args.min_samples_split
    max_depth=args.max_depth
    n_iter_no_change=args.n_iter_no_change


    cacheModel=args.cacheModel
    savedir=args.savedir
    repeat = args.repeat

    seed=random.randint(1,1000)

    if args.loadFeature==False:
        X_train, y_train = algebra_graph.bulidDataSet(Lin_train,etaE=i,kappa=j,etaL=i,v=l,A_matByE=A_matByE,L_matByE=L_matByE,A_matByL=A_matByL,L_matByL=L_matByL)
        X_val,y_val = algebra_graph.bulidDataSet(Lin_val,etaE=i,kappa=j,etaL=i,v=l,A_matByE=A_matByE,L_matByE=L_matByE,A_matByL=A_matByL,L_matByL=L_matByL)
        X_test,y_test = algebra_graph.bulidDataSet(Lin_test,etaE=i,kappa=j,etaL=i,v=l,A_matByE=A_matByE,L_matByE=L_matByE,A_matByL=A_matByL,L_matByL=L_matByL)
        feature_set_remain_20, label_set_remain_20 = algebra_graph.bulidDataSet(Li20,etaE=i,kappa=j,etaL=i,v=l,A_matByE=A_matByE,L_matByE=L_matByE,A_matByL=A_matByL,L_matByL=L_matByL)
        feature_set_remain_40, label_set_remain_40 = algebra_graph.bulidDataSet(Li40,etaE=i,kappa=j,etaL=i,v=l,A_matByE=A_matByE,L_matByE=L_matByE,A_matByL=A_matByL,L_matByL=L_matByL)
    else:
        index=args.index
        X_train = io_routine.loadCache("../pretreat/"+str(index)+"_X_train")
        y_train = io_routine.loadCache("../pretreat/"+str(index)+"_y_train")
        X_val = io_routine.loadCache("../pretreat/"+str(index)+"_X_val")
        y_val = io_routine.loadCache("../pretreat/"+str(index)+"_y_val")
        X_test = io_routine.loadCache("../pretreat/"+str(index)+"_X_test")
        y_test = io_routine.loadCache("../pretreat/"+str(index)+"_y_test")
        feature_set_remain_20 = io_routine.loadCache("../pretreat/"+str(index)+"_feature_set_remain_20")
        label_set_remain_20 = io_routine.loadCache("../pretreat/"+str(index)+"_label_set_remain_20")
        feature_set_remain_40 = io_routine.loadCache("../pretreat/"+str(index)+"_feature_set_remain_40")
        label_set_remain_40 = io_routine.loadCache("../pretreat/"+str(index)+"_label_set_remain_40")
    print("finish loading")

    X_train=data_norm(X_train)
    X_val=data_norm(X_val)
    X_test=data_norm(X_test)
    feature_set_remain_20=data_norm(feature_set_remain_20)
    feature_set_remain_40=data_norm(feature_set_remain_40)

    #X_train,X_val,X_test,feature_set_remain_20,feature_set_remain_40 = data_norm(X_train,X_val,X_test,feature_set_remain_20,feature_set_remain_40)

    print("--------finish guiyihua--------------")
    val_sum = np.zeros(y_val.shape) #存五次的测试结果，然后取平均值
    test_sum = np.zeros(y_test.shape)
    remain20_sum = np.zeros(label_set_remain_20.shape)
    remain40_sum = np.zeros(label_set_remain_40.shape)

    for m in range(repeat):
        if args.model=='GBR':
            model = train_GBR_model(X_train,y_train,n_estimators,subsample, min_samples_split, max_depth,seed,n_iter_no_change)
        elif args.model=='RFR':
            model = train_RFR_model(X_train, y_train,n_estimators, min_samples_split, max_depth,seed,continueTrain=False)
        elif args.model=='SVR':
            model=train_SVR_model(X_train,y_train,args.tol,args.epsilon,args.kernel,args.C,seed)
        #train_SVR_model(feature_set,label_set,tol,epsilon,kernel,C,seed)
        #def train_GBR_model(feature_set, label_set,n_estimators, subsample, min_samples_split, max_depth,seed,n_iter_no_change,continueTrain=False):
        if cacheModel:
            cacheData(model,"..//Pretrain//model"+"_"+args.model+"_"+str(m))
        val_sum = val_sum+model.predict(X_val)
        test_sum = test_sum+model.predict(X_test)
        remain20_sum = remain20_sum + model.predict(feature_set_remain_20)
        remain40_sum = remain40_sum + model.predict(feature_set_remain_40)
        seed=seed+1

    val_sum = val_sum / repeat
    print(type(val_sum))
    print(val_sum.shape)
    test_sum = test_sum / repeat
    remain20_sum = remain20_sum / repeat
    remain40_sum = remain40_sum / repeat

    MAE_val,MSE_val,RMSE_val,pccs_val,r2_val = evaluationResult(val_sum,y_val)
    MAE_test,MSE_test,RMSE_test,pccs_test,r2_test = evaluationResult(test_sum,y_test)
    MAE_20,MSE_20,RMSE_20,pccs_20,r2_20 = evaluationResult(remain20_sum,label_set_remain_20)
    MAE_40,MSE_40,RMSE_40,pccs_40,r2_40 = evaluationResult(remain40_sum,label_set_remain_40)

    fp=open(savedir,"a")
    fcntl.flock(fp, fcntl.LOCK_EX)
    writeLog("parameter:etaE,kappa,etaL,v",savedir)
    writeLog("{} {} {} {}".format(i,j,i,l),savedir)
    if args.model=='GBR':
        writeLog("n_estimators,subsample,min_sample,max_depth",savedir)
        writeLog("{} {} {} {}".format(str(n_estimators),str(subsample),str(min_samples_split),str(max_depth)),savedir)
    elif args.model=='RFR':
        writeLog("n_estimators,subsample,min_sample,max_depth",savedir) #RFR have no subsample, but for conviently deal log add it
        writeLog("{} {} {} {}".format(str(n_estimators),str(subsample),str(min_samples_split),str(max_depth)),savedir)
    elif args.model=='SVR':
        writeLog("tol,epsilon,kernel,C",savedir) #RFR have no subsample, but for conviently deal log add it
        writeLog("{} {} {} {}".format(str(args.tol),str(args.epsilon),args.kernel,str(args.C)),savedir)
    writeLog("result in validation",savedir)
    writeLog(str(MAE_val), savedir) 
    writeLog(str(MSE_val), savedir) 
    writeLog(str(RMSE_val), savedir) 
    writeLog(str(pccs_val[0]), savedir) 
    writeLog(str(r2_val), savedir) 

    writeLog("result in test",savedir)
    writeLog(str(MAE_test), savedir) 
    writeLog(str(MSE_test), savedir) 
    writeLog(str(RMSE_test), savedir) 
    writeLog(str(pccs_test[0]), savedir) 
    writeLog(str(r2_test), savedir) 

    writeLog("result in remain 20",savedir)
    writeLog(str(MAE_20), savedir) 
    writeLog(str(MSE_20), savedir) 
    writeLog(str(RMSE_20), savedir) 
    writeLog(str(pccs_20[0]), savedir) 
    writeLog(str(r2_20), savedir) 

    writeLog("result in remain 40",savedir)
    writeLog(str(MAE_40), savedir) 
    writeLog(str(MSE_40), savedir) 
    writeLog(str(RMSE_40), savedir) 
    writeLog(str(pccs_40[0]), savedir) 
    writeLog(str(r2_40), savedir) 

    writeLog("    ", savedir)
    fcntl.flock(fp, fcntl.LOCK_UN)
    fp.close()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args):

    
    parser=argparse.ArgumentParser(description="parameter")
    parser.add_argument('--eta',type=float,help='eta parameter in AGL')
    parser.add_argument('--kappa',type=float,help='eta parameter in AGL')
    parser.add_argument('--v',type=float,help='eta parameter in AGL')

    parser.add_argument('--n_estimators',type=int,help='parameter of ML')#for GBR & RFR model
    parser.add_argument('--subsample',type=float,help='parameter of ML')#for GBR model
    parser.add_argument('--min_samples_split',type=int,help='parameter of ML')#for GBR & RFR model
    parser.add_argument('--max_depth',type=int,help='parameter of ML')#for GBR & RFR model
    parser.add_argument('--n_iter_no_change',type=int,help='parameter of ML',default=None)#for GBR & RFR model

    parser.add_argument('--tol',type=float,help='parameter of ML')
    parser.add_argument('--epsilon',type=float,help='parameter of ML')
    parser.add_argument('--kernel',type=str,help='parameter of ML',choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
    parser.add_argument('--C',type=float,help='parameter of ML')

    parser.add_argument('--A_matByE',type=str2bool,help='whether using adjacency by general exponational',default=True)
    parser.add_argument('--L_matByE',type=str2bool,help='whether using Lap by general exponational',default=True)
    parser.add_argument('--A_matByL',type=str2bool,help='whether using adjacency by general Lor',default=True)
    parser.add_argument('--L_matByL',type=str2bool,help='whether using Lap by general Lor',default=True)


    parser.add_argument('--cacheModel',type=str2bool,help='whether saving models')
    parser.add_argument('--savedir',type=str,help='dir of the log')
    parser.add_argument('--repeat',type=int,help='repeat')

    parser.add_argument('--loadFeature',type=str2bool,help='loading feature generate by"build_pretratin.py" ',default=False)#used for turning ML para 
    parser.add_argument('--index',type=int,help='running model')#only for loadFeature


    parser.add_argument('--model',type=str,help='running model',choices=['GBR','RFR','SVR'],default='GBR')

    args=parser.parse_args()
    return args



def cli_main():
    args=parse_args(sys.argv[1:])
    print(args)
    main(args)

if __name__=="__main__":
    cli_main()
    print('End!')