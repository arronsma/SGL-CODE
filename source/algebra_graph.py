import kernelFunction
import numpy as np
import math
from CLUSTER import cluster

error0 = 1e-10 # zero for float

# <<<<<<<<<<< the function with "NP" at the end is implemented by numpy lib, it may be faster than the function without "np" if you dis_mat is a np.array <<<<<<<<<<<<
# I mean, may be
def DisMat2AdjacencyExponential(dis_mat): #abandon
#input: a n*n distance matrix
#output: a Adjacency using Exponential function
    n=dis_mat.shape[0]
    assert(dis_mat.shape[0]==dis_mat.shape[1])
    adjacency = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                adjacency[i][j] = 0
            else:
                adjacency[i][j] = kernelFunction.generalizedExponentialFunction(dis_mat[i][j],eta=1,kappa=1)
    return adjacency

def DisMat2AdjacencyExponentialNp_list(dis_mat,eta,kappa): 
#input: a n*n distance matrix
#output: a Adjacency list using Exponential function
#By definition a cluster with n atoms will have n adjacency_matrix
    n=dis_mat.shape[0]
    assert(dis_mat.shape[0]==dis_mat.shape[1])
    kDis_mat = kernelFunction.generalizedExponentialFunctionNp(dis_mat,eta=eta,kappa=kappa)
    for i in range(n):
        kDis_mat[i][i]=0
    adjacency_mat_list = []
    for i in range(n):
        adjacency_mat = np.zeros((n,n))
        adjacency_mat[i,:] = kDis_mat[i,:]
        adjacency_mat[:,i] = kDis_mat[:,i]
        adjacency_mat_list.append(adjacency_mat)
    return adjacency_mat_list

#<<<<<<<< A2Lfunction is for Exponential and Lorentz fucntion; 
# if the input adjacency_mat is using Exponential function, the output L_mat is using Exponential function
# if the input adjacency_mat is using Lorentz function, the output L_mat is using Exponential function <<<<<<<<<<

def Adjacency2Laplacian (adjacency_mat):#abandon
    # input : a Adjacency using Exponential function
    # output: a Laplacian mat using Exponential function
    # only modify diagnol
    Laplacian_mat = -np.array(adjacency_mat)
    assert(id(adjacency_mat)!=id(Laplacian_mat))
    sum = np.sum(Laplacian_mat,axis=0)
    for i in range(adjacency_mat.shape[0]):
        Laplacian_mat[i][i] = -sum[i]
    return Laplacian_mat

def Adjacency_list2Laplacian_list(adjacency_mat_list):
    laplacian_mat_list = []
    for i in range(len(adjacency_mat_list)):
        adjacency_mat = adjacency_mat_list[i]
        laplacian_mat = Adjacency2Laplacian(adjacency_mat)
        laplacian_mat_list.append(laplacian_mat)
    return laplacian_mat_list

#<<<<<<<<<<<<<<<<<<<<<<<<<<<< using Lorentz fucntion to construct A mat and L_mat<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def DisMat2AdjacencyLorentz(dis_mat): 
#input: a n*n distance matrix
#output: a Adjacency using Lorentz function
    n=dis_mat.shape[0]
    assert(dis_mat.shape[0]==dis_mat.shape[1])
    adjacency = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                adjacency[i][j] = 0
            else:
                adjacency[i][j] = kernelFunction.generalizedLorentzFunction(dis_mat[i][j],eta=1,v=1)
    return adjacency

def DisMat2AdjacencyLorentzNp_list(dis_mat,eta,v): 
#input: a n*n distance matrix
#output: a Adjacency list using Lorentz function； len(list) = n
    n=dis_mat.shape[0]
    assert(dis_mat.shape[0]==dis_mat.shape[1])
    kDis_mat = kernelFunction.generalizedLorentzFunctionNp(dis_mat,eta=eta,v=v)
    for i in range(n):
        kDis_mat[i][i]=0
    adjacency_mat_list = []
    for i in range(n):
        adjacency_mat = np.zeros((n,n))
        adjacency_mat[i,:] = kDis_mat[i,:]
        adjacency_mat[:,i] = kDis_mat[:,i]
        adjacency_mat_list.append(adjacency_mat)
    return adjacency_mat_list

# <<<<<<<<<<<<<<<<<<<<<<<<<< now is going to build feature of only a dis_mat of an cluster <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def mat2nineFeature(mat,typeM):
    #input:a matrix, type
    #output: a list of 9 feature
    #getting 9 feature of 4 matrices, total 36 features
    #       one can compute nine
    #   >>> descriptive statistical values, namely the sum, minimum (i.e.,
    #   >>> the Fiedler value for Laplacian matrices or the half band gap
    #   >>> for adjacency matrices), maximum, mean, median, standard
    #   >>> deviation, and variance of all eigenvalues.
    assert(typeM == "adjacency" or typeM == "laplacian")
    eigenvalue,b=np.linalg.eig(mat) # a is eigenvalue, egenvalue is a list , maybe it should be named egenvalues. 
    # n_origin = eigenvalue.shape[0]
    b # if i don't use variable b, vscode will give me a warn
    if (typeM=="adjacency") :# for adjacency matrix, we only consider positive eigenvalue
        eigenvalue[eigenvalue<= error0]= 0 # some zeros
        eigenvalue = np.ma.masked_equal(eigenvalue,0)
        eigenvalue=np.ma.compressed(eigenvalue)
        if eigenvalue.shape[0] == 0:
            print("warning: A mat egi all zero")
            return [0,0,0,0,0,0,0,0,0]
        # assert(math.ceil(n_origin-1)/2 == eigenvalue.shape[0]) 
        # ERROR: because the eigenvalue is symmetry by 0 point. if the number of original eigenvalues is odd (eg.3), it will remain 1 eigenvalue which is positive (1positive 1negative 1 is zero)
        # if the number of original eigenvalues is even(eg. 10), it will remain 5   
        # REASON: because there can be more than one element that is zero
    Fsum=np.sum(eigenvalue)
    if typeM == "adjacency":
        Fminimum = np.min(eigenvalue)
        assert(Fminimum>error0) # here shoudn't be non-positive number
    elif typeM == "laplacian":
        Fminimum = 100000
        for i in range(len(eigenvalue)):
            if eigenvalue[i] < Fminimum and eigenvalue[i]>error0:
                Fminimum = eigenvalue[i]
        if Fminimum>9999:
            print("warning: L mat egi all zero")
            return [0,0,0,0,0,0,0,0,0]
    Fmaximum = np.max(eigenvalue)
    Fmean = np.mean(eigenvalue)
    Fmedian = np.median(eigenvalue)
    #numpy.std() 求标准差的时候默认是除以 n 的，即是有偏的，np.std无偏样本标准差方式为加入参数 ddof = 1；
    Fstd = np.std(eigenvalue)
    Fvar = np.var(eigenvalue)
    Fnum = eigenvalue.shape[0]
    Fsum2 = np.sum(np.power(eigenvalue,2))
    return [Fsum,Fminimum,Fmaximum,Fmean,Fmedian,Fstd,Fvar,Fnum,Fsum2]

def nineStatistic(feature):#because the len of feature depend on the n_atoms, so statistic it again
    Fsum=np.sum(feature)
    Fminimum=np.min(feature)
    Fmaximum=np.max(feature)
    Fmean=np.mean(feature)
    Fmedian = np.median(feature)
    Fstd = np.std(feature)
    Fvar = np.var(feature)
    Fnum = len(feature)
    Fsum2= np.sum(np.power(feature,2))
    return [Fsum,Fminimum,Fmaximum,Fmean,Fmedian,Fstd,Fvar,Fnum,Fsum2]

def DisMat2featureVector (dis_mat,etaE,kappa,etaL,v,A_matByE=True,L_matByE=True,A_matByL=True,L_matByL=True):
    # input: a dis_mat
    # input: etaE and kappa for Exponential fucntion
    # input: etaL and v for Lorentz function
    # 1. geting 4 matrices: 
    #   A_matByE_list : Adjacency matrix using Exponential
    #   L_matByE_list
    #   A_matByL_list
    #   L_matByL_list
    # 2. getting 9 feature of all matrices in  4 matrices_list, total 36 features
    #       one can compute nine
    #   >>> descriptive statistical values, namely the sum, minimum (i.e.,
    #   >>> the Fiedler value for Laplacian matrices or the half band gap
    #   >>> for adjacency matrices), maximum, mean, median, standard
    #   >>> deviation, and variance of all eigenvalues.
    # 
    #     we also utilize the number of eigenvalues and the sum of the second
    # >>> power of eigenvalues
    # updata:modifying the last 4 parameters can modify the composition of feature.
    
    A_matByE_list = DisMat2AdjacencyExponentialNp_list(dis_mat,eta=etaE,kappa=kappa)
    L_matByE_list = Adjacency_list2Laplacian_list(A_matByE_list)
    A_matByL_list = DisMat2AdjacencyLorentzNp_list(dis_mat,eta=etaL,v=v)
    L_matByL_list = Adjacency_list2Laplacian_list(A_matByL_list)

    n_atom = dis_mat.shape[0]
    feature1=[] #all features of adjacency matrix by Exponential --- there is n*9 feature n is the number of atoms
    feature2=[] #all features of laplacian matrix by Exponential --- there is n*9 feature n is the number of atoms
    feature3=[] #all features of adjacency matrix by Lorentz --- there is n*9 feature n is the number of atoms
    feature4=[] #all features of laplacian matrix by Lorentz --- there is n*9 feature n is the number of atoms
    for i in range(n_atom):
        feature1 = feature1+mat2nineFeature(A_matByE_list[i],typeM="adjacency")
    for i in range(n_atom):
        feature2 = feature2+mat2nineFeature(L_matByE_list[i],typeM="laplacian")
    for i in range(n_atom):
        feature3 = feature3+mat2nineFeature(A_matByL_list[i],typeM="adjacency")
    for i in range(n_atom):
        feature4 = feature4+mat2nineFeature(L_matByL_list[i],typeM="laplacian")

    statisticFeature=[]
    if A_matByE:
        statisticFeature = statisticFeature + nineStatistic(feature1)
    if L_matByE:
        statisticFeature = statisticFeature + nineStatistic(feature2)
    if A_matByL:
        statisticFeature = statisticFeature + nineStatistic(feature3)
    if L_matByL:
        statisticFeature = statisticFeature + nineStatistic(feature4)
    return statisticFeature

def bulidDataSet(datasetIn: cluster,etaE,kappa,etaL,v,A_matByE=True,L_matByE=True,A_matByL=True,L_matByL=True):
    feature_set = []
    label_set = []
    times = 0
    for i in range(len(datasetIn)):
        feature_set.append(DisMat2featureVector(datasetIn[i].dis_mat,etaE=etaE,kappa=kappa,etaL=etaL,v=v,A_matByE=A_matByE,L_matByE=L_matByE,A_matByL=A_matByL,L_matByL=L_matByL))
        label_set.append(datasetIn[i].energy)
        times = times+1
        if times % 10000 == 0:
            print("convert {} cluster".format(times))
    feature_set = np.array(feature_set,dtype='float64')
    label_set = np.array(label_set,dtype='float64')
    return feature_set,label_set
    

