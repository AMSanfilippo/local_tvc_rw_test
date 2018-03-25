#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:14:03 2018

@author: as
"""

import pandas as pd
import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mpld

from clean_data import *
from residual_bootstrap import *


# gaussian kernel weights
# tau: target time value on [0,1]
# times: set of all regularized times, from 0 to 1
# h: bandwidth as proportion of total data
def gauss(tau,times,h):
    time_diffs = np.subtract(times,tau)
    kernel_inputs = np.divide(time_diffs,h)
    kernel_outputs = scipy.stats.norm.pdf(kernel_inputs)
    wts = np.divide(kernel_outputs,h)
    return wts

def wls(X,Y,W):
    XT_W = (X.T).dot(W) 
    # B_hat = inv(X'WX) * (X'WY)
    B_hat = np.linalg.solve(XT_W.dot(X),XT_W.dot(Y)) # faster than inversion
    # return coefficient estimates and residuals
    return B_hat.T

kernels = {
  
        'gaussian':gauss
        
}

data = clean_data('19940101','20031230') 

asset = 'Durbl'
model = 'CAPM' # 'FF3'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,model)
# test FF3 specification, assuming lag-one AR in returns
# X = gen_X(data,asset,'FF3')
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T) # excess returns

# iterate through normalized dates
normalized_datelist = np.divide(range(len(Y)),len(Y))

coeffs = pd.read_csv('output/Durbl/nonparam_coeff_ests_CAPM.csv')
coeffs = coeffs[['const','rm_rf','r_1']]

h = 0.014

wtmat = np.matrix([0]*len(normalized_datelist))
for tau in normalized_datelist:
    outwts = np.asmatrix(gauss(tau,normalized_datelist,h))
    wtmat = np.concatenate((wtmat,outwts))

wtmat = wtmat[1:,:] # ith row = wts. for estimating ith set of coeffs.

W = np.zeros((len(Y),len(Y)),float)

centered_resids = get_residuals('alternative',X,Y,coeffs,model)['centered_resids'] 
   
r_0 = X[0,-1] # lagged return value to serve as "seed" for DGP
X_cp = X.copy() # X matrix for DGP

B = 5 # number of bootstrap replications
r_1_bs = pd.DataFrame(np.zeros((len(Y),B),float))
for i in range(B):
    print('bs iteration: ', (i+1))
    bs_resids = bs_resample(centered_resids.values)
    recursive = gen_recursive('alternative',X_cp,r_0,bs_resids,coeffs,model)
    X_cp[:,-1] = recursive[0]
    Y_recur = recursive[1]
    results = np.matrix([0]*len(X.T))
    # estimate nonparametric regression on bootstrap dataset
    for d in range(len(Y_recur)):
        np.fill_diagonal(W,wtmat[d]) # diagonal matrix of weights
        out = wls(X_cp,Y_recur,W)
        # return coefficient estimates and residuals
        results = np.concatenate((results,out))
    r_1_bs[i] = results[1:,-1]

