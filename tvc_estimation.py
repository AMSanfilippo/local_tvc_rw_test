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
from joblib import Parallel, delayed
import multiprocessing

from clean_data import *
from residual_bootstrap import *

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

# symmetric uniform weights (until Gauss kernel is fixed)
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


# WLS regression
# X: independent variable matrix with column of 1s for intercept
# Y: dependent variable matrix
# target: target time on [1...n], passed as implicit index starting at 0
# h: bandwidth (proportion on [0,1])
def wls(X,Y,target,h,wt_fxn):
    print(target)
    W = np.diag(kernels[wt_fxn](target,h,list(range(len(Y))))) # diagonal matrix of weights
    # B_hat = inv(X'WX) * (X'WY)
    B_hat = np.linalg.inv(X.T*W*X)*(X.T*W*Y) 
    # get residuals
    resids = Y - (X*B_hat)
    # return coefficient estimates and residuals
    return B_hat.flatten().tolist()[0]


###############################################################################

numcores = multiprocessing.cpu_count()

data = clean_data('19940101','20031230')

asset = 'Telcm'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,'CAPM')
Y = (np.matrix(data.loc[1:,asset].values).T)

n = len(Y)
h_test = 0.1
start = math.floor((h*n)/2)
end = n - start

datelist = list(range(len(Y)))[start:end]

# run for loop in parallel
# with 8 cores and about 10 years of data, this runs in ~ 25 secs
# --> 500 bs replications should take ~ 35 mins
results = Parallel(n_jobs=numcores)(delayed(wls)(X=X,Y=Y,target=date,h=h_test,wt_fxn='uniform') for date in datelist)

coeffs = pd.DataFrame(results)
coeffs.columns = ['const','rm_rf','r_1']

# plot fitted values
fitted = get_residuals('alternative',X,Y,coeffs,'CAPM',start,end)['fitted']

x = range(1,len(datelist_test)+1)

fig, ax = plt.subplots()

line1, = ax.plot(x[:100], fitted[:100].values, dashes = [5,5], linewidth=2,color='b',label='fitted')
line2, = ax.plot(x[:100], Y[start:start+100], linewidth=2,color='r')

plt.show()

###############################################################################

centered_resids = get_residuals('alternative',X,Y,coeffs,'CAPM',start,end)['centered_resids']

bs_resids = bs_resample(centered_resids.values)

recursive_test = gen_recursive('alternative',coeffs,X.copy(),bs_resids,'CAPM',start,end)

X_recursive = recursive_test[0] # full X-matrix, with [start+1:end,'r_1'] replaced with recursively-generated outcomes
Y_recursive = Y.copy()
Y_recursive[start:end] = recursive_test[1] # full Y-matrix, with [start:end] replaced with recursively-generated outcomes

x = range(len(Y[start:start+100]))

fig, ax = plt.subplots()

line1, = ax.plot(x, Y[start:start+100], linewidth=2,color='b')
line2, = ax.plot(x, Y_recursive[start:start+100], dashes = [5,5], linewidth=2,color='r')

plt.show()