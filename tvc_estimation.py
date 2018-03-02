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


# gaussian kernel weights
# tau: target time value on [0,1]
# times: set of all regularized times, from 0 to 1
# h: bandwidth as proportion of total data
def gauss(tau,times,h):
    kernel_inputs = [(t-tau)/h for t in times]
    kernel_outputs = [(1/math.sqrt(2*math.pi))*np.exp(-(i**2)/2) for i in kernel_inputs]
    wts = [k/h for k in kernel_outputs]
    return wts

        
# kernels to choose from. 
# right now just uniform
kernels = {
  
        'gaussian':gauss
        
}


# WLS regression
# X: independent variable matrix with column of 1s for intercept
# Y: dependent variable matrix
# target: target time on [1...n], passed as implicit index starting at 0
# h: bandwidth (proportion on [0,1])
def wls(X,Y,target,wt_fxn,h):
    print(target)
    tau = target/len(Y) # normalize target value on [0,1]
    times = [i/len(Y) for i in list(range(len(Y)))]
    W = np.diag(kernels[wt_fxn](tau,times,h)) # diagonal matrix of weights
    # B_hat = inv(X'WX) * (X'WY)
    B_hat = np.linalg.inv(X.T*W*X)*(X.T*W*Y) 
    # return coefficient estimates and residuals
    return B_hat.flatten().tolist()[0]


###############################################################################

numcores = multiprocessing.cpu_count()

data = clean_data('19940101','20031230')

asset = 'Telcm'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,'CAPM')
Y = (np.matrix(data.loc[1:,asset].values).T)

datelist = list(range(len(Y)))

# run for loop in parallel
# with 8 cores and about 10 years of data, this runs in ~ 25 secs
# --> 500 bs replications should take ~ 35 mins
results = Parallel(n_jobs=numcores)(delayed(wls)(X=X,Y=Y,target=date,wt_fxn='gaussian',h=h_test) for date in datelist)

coeffs = pd.DataFrame(results)
coeffs.columns = ['const','rm_rf','r_1']

# plot fitted values
fitted = get_residuals('alternative',X,Y,coeffs,'CAPM')['fitted']

x = range(len(datelist_test))

fig, ax = plt.subplots()

line1, = ax.plot(x[:100], fitted[:100].values, dashes = [5,5], linewidth=2,color='b',label='fitted')
line2, = ax.plot(x[:100], Y[:100], linewidth=2,color='r')

plt.show()

###############################################################################

datelist = list(range(len(Y)))

centered_resids = get_residuals('alternative',X,Y,coeffs,'CAPM')['centered_resids']

const_bs = pd.DataFrame()
rm_rf_bs = pd.DataFrame()
r_1_bs = pd.DataFrame()

B = 5 # number of bootstrap replications
for i in range(B):
    print('bs iteration: ', i)
    bs_resids = bs_resample(centered_resids.values)
    X_cp = X.copy()
    recursive = gen_recursive('alternative',coeffs,X_cp,bs_resids,'CAPM')
    X_recur = recursive[0]
    Y_recur = recursive[1]
    bs_est = Parallel(n_jobs=numcores)(delayed(wls)(X=X_recur,Y=Y_recur,target=date,wt_fxn='gaussian',h=h_test) for date in datelist)
    bs_est_mat = np.asmatrix(bs_est)
    const_bs[i] = bs_est_mat[:,0].flatten().tolist()[0]
    rm_rf_bs[i] = bs_est_mat[:,1].flatten().tolist()[0]
    r_1_bs[i] = bs_est_mat[:,2].flatten().tolist()[0]
    
