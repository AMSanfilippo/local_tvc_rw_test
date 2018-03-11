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


def ols(X,Y):
    B_hat = np.linalg.inv(X.T*X)*(X.T*Y)
    return B_hat.flatten().tolist()[0]

###############################################################################

numcores = multiprocessing.cpu_count()

data = clean_data('19940101','20031230')

asset = 'Telcm'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,'CAPM')
Y = (np.matrix(data.loc[1:,asset].values).T)

datelist = list(range(len(Y)))

h = 0.1

# run for loop in parallel
# with 8 cores and about 10 years of data, this runs in ~ 25 secs
# --> 500 bs replications should take ~ 35 mins
results = Parallel(n_jobs=numcores)(delayed(wls)(X=X,Y=Y,target=date,wt_fxn='gaussian',h=h) for date in datelist)

coeffs = pd.DataFrame(results)
coeffs.columns = ['const','rm_rf','r_1']

# plot fitted values
fitted = get_residuals('alternative',X,Y,coeffs,'CAPM')['fitted']

x = range(len(datelist))

fig, ax = plt.subplots()

line1, = ax.plot(x[:50], fitted[:50].values, dashes = [5,5], linewidth=2,color='b',label='fitted')
line2, = ax.plot(x[:50], Y[:50], linewidth=2,color='r')

plt.show()

###############################################################################

datelist = list(range(len(Y)))

# centered_resids = get_residuals('alternative',X,Y,coeffs,'CAPM')['centered_resids']
resids = get_residuals('alternative',X,Y,coeffs,'CAPM')['resids'] # probably don't need to use centered resids

const_bs = pd.DataFrame()
rm_rf_bs = pd.DataFrame()
r_1_bs = pd.DataFrame()

B = 10 # number of bootstrap replications
for i in range(B):
    print('bs iteration: ', i)
    bs_resids = bs_resample(resids.values)
    X_cp = X.copy()
    recursive = gen_recursive('alternative',X_cp,bs_resids,coeffs,'CAPM')
    X_recur = recursive[0]
    Y_recur = recursive[1]
    bs_est = Parallel(n_jobs=numcores)(delayed(wls)(X=X_recur,Y=Y_recur,target=date,wt_fxn='gaussian',h=h) for date in datelist)
    bs_est_mat = np.asmatrix(bs_est)
    const_bs[i] = bs_est_mat[:,0].flatten().tolist()[0]
    rm_rf_bs[i] = bs_est_mat[:,1].flatten().tolist()[0]
    r_1_bs[i] = bs_est_mat[:,2].flatten().tolist()[0]
 
    
r_1_hat = coeffs['r_1'] # point estimates for lag-one coefficient
bs_sd = np.std(r_1_bs,axis=1,ddof=1)
Q = (r_1_bs.sub(r_1_hat,axis=0)).divide(bs_sd,axis=0) # Q = (r_1_hat - r_1*)/sd(r_1*) for all t

# empirical critical value for 95% pointwise CI
c = np.percentile(Q,97.5,axis=1,interpolation='higher') # may want to think about different interpolation param
me =  [a*b for a,b in zip(bs_sd.values,c)]# margin of error

# CIs: b_hat +/- c*sd for each t
CI_lb = [a - b for a,b in zip(r_1_hat.values,me)]
CI_ub = [a + b for a,b in zip(r_1_hat.values,me)]

# plot 95% pointwise CIs
x = range(len(datelist))

fig, ax = plt.subplots()

line1, = ax.plot(x, CI_lb, dashes = [5,5], linewidth=2,color='b')
line2, = ax.plot(x, CI_ub, dashes = [5,5], linewidth=2,color='b')
line3, = ax.plot(x, r_1_hat.values, linewidth=2,color='r')
line4, = ax.plot(x,[0]*len(x),dashes = [5,5],color='grey')

plt.show()

###############################################################################

ols_out = ols(X,Y) 

null_resids = get_residuals('null',X,Y)['resids']

rss_restricted = sum([a**2 for a in null_resids.values])

alternative_resids = get_residuals('alternative',X,Y,coeffs,'CAPM')['resids'] 
rss_unrestricted = sum([a**2 for a in alternative_resids.values])  

tau_hat = (rss_restricted - rss_unrestricted)/rss_unrestricted

B = 10 # number of bootstrap replications
taus = [0]*B
alternative_resids_centered = get_residuals('alternative',X,Y,coeffs,'CAPM')['centered_resids'] # should this be centered or not??

# compute null distribution of the test statistic using bootstrap
for i in range(B):
    print('bs iteration: ', i)
    bs_resids = bs_resample(alternative_resids_centered.values) 
    X_cp = X.copy()
    recursive = gen_recursive('null',X_cp,bs_resids,ols_out)
    X_recur = recursive[0]
    Y_recur = recursive[1]
    bs_alt_est = Parallel(n_jobs=numcores)(delayed(wls)(X=X_recur,Y=Y_recur,target=date,wt_fxn='gaussian',h=h) for date in datelist)
    bs_alt_est_coeffs = pd.DataFrame(bs_alt_est)
    bs_alt_est_coeffs.columns = ['const','rm_rf','r_1']
    bs_alt_resids = get_residuals('alternative',X_recur,Y_recur,bs_alt_est_coeffs,'CAPM')['resids']
    bs_alt_rss = sum([a**2 for a in bs_alt_resids.values])  
    # it's not entirely clear based on sources what one is meant to do here:
    # it seems one would calculate null RSS in the BS sample, and use this for test statistic
    bs_null_resids = get_residuals('null',X_recur,Y_recur)['resids']
    bs_null_rss = sum([a**2 for a in bs_null_resids.values])
    taus[i] = (bs_null_rss - bs_alt_rss)/bs_alt_rss # T* in null distribution


