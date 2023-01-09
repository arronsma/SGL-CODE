import numpy as np
import math
#function with Np is for numpy matrix, without Np is for a real number
def generalizedExponentialFunction (r,eta=2.0,kappa=3.0): #r is the Euclidean space distance; eta and kappa is two parameter of the function
    return math.exp(-math.pow(r/eta,kappa))

def generalizedLorentzFunction (r,eta=2.0,v=3.0): # r is the distance, eta and v is 2 parameters
    return 1/(1+pow(r/eta, v))

def generalizedExponentialFunctionNp (r_mat,eta=2.0,kappa=3.0): #r is the Euclidean space distance_matrix; eta and kappa is two parameter of the function
    return np.exp(-np.power(r_mat/eta,kappa))

def generalizedLorentzFunctionNp (r_mat,eta=2.0,v=3.0): # r_mat is the distance_matrix, eta and v is 2 parameters 
    return 1/(1+np.power(r_mat/eta, v))