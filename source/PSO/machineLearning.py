from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
import numpy as np
import os
from sklearn import model_selection
from sklearn import metrics
from scipy.stats import pearsonr

from sklearn import model_selection

import random
import io_routine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR as LSVR
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def train_GBR_model(feature_set, label_set,n_estimators, subsample, min_samples_split, max_depth,seed,continueTrain=False): #计划加入一个savedir参数保存模型
    #input:
    #   training data; should be numpy
    #output: 
    #   a model after training
    print(feature_set.shape)
    model=GBR(loss='ls',learning_rate=0.001,n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,subsample=subsample,max_features='sqrt',random_state=seed)
    history = model.fit(feature_set,label_set) #this GBR has not validation set. if you want to test the model using test set, you should call another function
    # model.save("./model_cache/model.h5")
    # drawing_losses_after_train(history)
    return model

def train_RFR_model(feature_set, label_set,n_estimators, min_samples_split, max_depth,seed,continueTrain=False): 
    print("start training RFR model")
    model=RFR(max_features='sqrt',n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,random_state=seed,verbose=1,n_jobs=-1)
    history=model.fit(feature_set,label_set)
    return model
    
def train_SVR_model(feature_set,label_set,tol,epsilon,seed):
    print("start training SVR model")
    model=SVR(verbose=false,tol=tol,epsilon=epsilon)
    model.fit(feature_set,label_set)
    return model

def train_LSVR_model(feature_set,label_set,seed):
    print("start training LSVR model")
    model=LSVR(verbose=1,random_state=seed)
    model.fit(feature_set,label_set)
    return model

def test_dense_model(feature_test, label_test,model):
    #input:
    #   A model,and the test data
    #output:
    #   MAE of the model undering the test data
    
    feature_test = np.array(feature_test,dtype='float32')
    predict = model.predict(feature_test)
    y_hat = np.array(predict,dtype='float')
    y_true = np.array(label_test,dtype='float').reshape(-1,1)

    MAE = metrics.mean_absolute_error(y_true,y_hat)
    MSE = metrics.mean_squared_error(y_true,y_hat)
    RMSE = MSE**0.5
    y_hat=np.squeeze(y_hat)
    y_true=np.squeeze(y_true)
    pccs = pearsonr(y_hat,y_true)
    io_routine.writeLog("性能","../log/logSum")
    io_routine.writeLog(str(MAE),"../log/logSum")
    io_routine.writeLog(str(MSE),"../log/logSum")
    io_routine.writeLog(str(RMSE),"../log/logSum")
    io_routine.writeLog(str(pccs),"../log/logSum")

def evaluationGBR(model,X_test,y_test):
    X_test = np.array(X_test,dtype='float32')
    y_test = np.array(y_test,dtype='float32')
    pred_test = model.predict(X_test)
    MAE = metrics.mean_absolute_error(y_test,pred_test)
    MSE = metrics.mean_squared_error(y_test,pred_test)
    RMSE = MSE**0.5
    pccs = pearsonr(y_test,pred_test)
    r2 = metrics.r2_score(y_test, pred_test)
    return MAE,MSE,RMSE,pccs,r2

def evaluationResult(y_hat,y_true):
    #input y_hat and y_true
    MAE = metrics.mean_absolute_error(y_true,y_hat)
    MSE = metrics.mean_squared_error(y_true,y_hat)
    RMSE = MSE**0.5
    pccs = pearsonr(y_true,y_hat)
    r2 = metrics.r2_score(y_true, y_hat)
    return MAE,MSE,RMSE,pccs,r2

def standardor(x):
    ss=StandardScaler()
    ss.fit(x)
    x=ss.fit_transform(x)
    return x

    
def data_norm(*args):
    assert len(args)>0,"datasets' length needs > 0" #会把输入参数变成一个元组，元组的一个元素是一个输入参数
    scaler=StandardScaler()
    scaler.fit(np.vstack(args))
    norm_args = [scaler.transform(args[i]) for i in range(len(args))]
    norm_args = norm_args if (len(args)>1 )else norm_args[0]
    return norm_args

def plot_distribution (pred_test, label_test,filename="../model_cache/model.h5"):
    #input:
    #   pred_test given by the model; label of the test set
    #output:
    #   A picture of the scatter of energy_predict and energy_label
    plt.scatter(pred_test,label_test)
    plt.xlabel("DFT binding energy")
    plt.ylabel("ML prediction energy")
    line = min(pred_test.min(),label_test.min())
    line_max = max(pred_test.max(),label_test.max())
    plt.plot([line,line_max],[line,line_max],'r')
    plt.show()
    plt.savefig("../scatter.jpg")
