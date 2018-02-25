#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:48:15 2018

@author: as
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from clean_data import *

"""
# Attempt at gauss kernel, the results makes no sense
# Kernel weights don't sum to 1?

def gauss(a):
    return (1/math.sqrt(2*math.pi))*np.exp(-(a**2)/2)

times = list(range(100))
h = 0.1
n = len(times)
for t in times[1:99]:
    timediffs = [(((i/n) - (t/n))/h) for i in times]
    print(sum(timediffs))
    testwts = [gauss(d)/h for d in timediffs]
    print(sum(testwts))
"""

# symmetric uniform weights until Gauss kernel is fixed
# t: target time, passed as implicit index starting at 0
# h: bandwidth (proportion on [0,1])
# times: list of t on [1...n]
def unif(t,h,times):
    interval = math.floor((h*len(times))/2)
    wts = [0]*len(times)
    wts[t - interval:t + interval + 1] = [1/len(wts[t - interval:t + interval + 1])]*len(wts[t - interval:t + interval + 1])
    return wts
        
# kernels to choose from. 
# right now just uniform
kernels = {
  
        'uniform':unif
        
}


# function to perform WLS regression
# X: independent variable matrix with column of 1s for intercept
# Y: dependent variable matrix
# target: target time on [1...n], passed as implicit index starting at 0
# h: bandwidth (proportion on [0,1])
def wls(X,Y,target,h,wt_fxn):

    W = np.diag(kernels[wt_fxn](target,h,list(range(len(Y))))) # diagonal matrix of weights
    # B_hat = inv(X'WX) * (X'WY)
    B_hat = np.linalg.inv(X.T*W*X)*(X.T*W*Y) 
    # get residuals
    resids = Y - (X*B_hat)
    # return coefficient estimates and residuals
    return [B_hat,resids]


# rolling kernel-weighted regression
def rolling_regression(X,Y,h,wt_fxn):

    # keep track of (time-varying) coefficient estimates and estimators for asymptotic variance
    coeffs = pd.DataFrame()
    # resids = [] 

    n = len(Y)
    start = math.floor((h*n)/2)
    end = n - start

    # run weighted local regression 
    for target in list(range(len(Y)))[start:end]: # center window around target value. use implicit index starting at 0
        print(target)
        # run wls, centering weights at target observation
        resp = wls(X,Y,target,h,wt_fxn) # run wls regression. return coefficient estimates and estimators
        
        coeff_sub = pd.DataFrame([resp[0].flatten().tolist()[0]])
        coeffs = coeffs.append(coeff_sub)

        #resids.append(resp[1].flatten().tolist()[0]) 

    return coeffs
    #return [coeffs,resids,Ws] 

###############################################################################

data = clean_data('19800101','20161230')

# test CAPM specification, assuming lag-one AR in returns
asset = 'NoDur'

X = gen_X(data,asset,'CAPM')
Y = np.matrix(data.loc[1:,asset].values).T

reg_out = rolling_regression(X,Y,0.2,'uniform')

# This works, BUT IT RUNS SUPER SLOWLY.
# It can't even get through all 9000+ data points. This will certainly NOT work for bootstrapping.
# Need to figure out some way to run this faster (in parallel?)
