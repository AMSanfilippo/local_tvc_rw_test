#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:48:15 2018

@author: as
"""

import pandas as pd
import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt

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

        
# kernels to choose from. 
# right now just uniform
kernels = {
  
        'gaussian':gauss
        
}


# WLS regression
# X: independent variable matrix with column of 1s for intercept
# Y: dependent variable matrix
# W: diagonal matrix of kernel weights
def wls(X,Y,W):
    XT_W = (X.T).dot(W) 
    # B_hat = inv(X'WX) * (X'WY)
    B_hat = np.linalg.solve(XT_W.dot(X),XT_W.dot(Y)) # faster than inversion
    # return coefficient estimates and residuals
    return B_hat.T

# OLS regression
# X: independent variable matrix with column of 1s for intercept
# Y: dependent variable matrix
def ols(X,Y):
    B_hat = np.linalg.inv(X.T*X)*(X.T*Y)
    return B_hat.flatten().tolist()[0]

###############################################################################

data = clean_data('19940101','20031230') 

asset = 'Telcm'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,'CAPM')
# test FF3 specification, assuming lag-one AR in returns
# X = gen_X(data,asset,'FF3')
Y = (np.matrix(data.loc[1:,asset].values).T)

# iterate through normalized dates
normalized_datelist = np.divide(range(len(Y)),len(Y))

h = 0.1 # bandwidth

# to speed up regression: compute and store kernel weights first
# note: weights are f(h,n) only
# a given bootstrap procedure will always use the same h and n, so this is valid
wtmat = np.matrix([0]*len(normalized_datelist))
for tau in normalized_datelist:
    outwts = np.asmatrix(gauss(tau,normalized_datelist,h))
    wtmat = np.concatenate((wtmat,outwts))

wtmat = wtmat[1:,:] # ith row = wts. for estimating ith set of coeffs.

W = np.zeros((len(Y),len(Y)),float) # empty, to become diagonal weight matrix
results = np.matrix([0]*len(X.T))

# estimate nonparametric regression
# 8 SECONDS!!
for d in range(len(Y)):
    np.fill_diagonal(W,wtmat[d]) # diagonal matrix of weights
    out = wls(X,Y,W)
    # return coefficient estimates and residuals
    results = np.concatenate((results,out))

coeffs = pd.DataFrame(results[1:,:])
coeffs.columns = ['const','rm_rf','r_1']
# coeffs.columns = ['const','rm_rf','smb','hml','r_1']

# plot fitted values
# fitted = get_residuals('alternative',X,Y,coeffs,'FF3')['fitted']
fitted = get_residuals('alternative',X,Y,coeffs,'CAPM')['fitted']

x = range(len(normalized_datelist))

fig, ax = plt.subplots()

line1, = ax.plot(x[:50], fitted[:50].values,color='b',dashes=[5,5])
line2, = ax.plot(x[:50], Y[:50], linewidth=2,color='r')

plt.show()

###############################################################################

# NOTE: use same normalized_datelist, h, and wtmat/W as in the above estimation

centered_resids = get_residuals('alternative',X,Y,coeffs,'CAPM')['centered_resids'] 
# resids = get_residuals('alternative',X,Y,coeffs,'FF3')['centered_resids'] 
   
r_0 = X[0,-1] # lagged return value to serve as "seed" for DGP
X_cp = X.copy() # X matrix for DGP

B = 100 # number of bootstrap replications
r_1_bs = pd.DataFrame(np.zeros((len(Y),B),float))
for i in range(B):
    print('bs iteration: ', (i+1))
    bs_resids = bs_resample(centered_resids.values)
    recursive = gen_recursive('alternative',X_cp,r_0,bs_resids,coeffs,'CAPM')
    # recursive = gen_recursive('alternative',X_cp,r_0,bs_resids,coeffs,'FF3')
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
 
    
### ### CURRENT STOPPING POINT FOR OPTIMIZATION ### ### 
# Save bootstrap output FFR, so that we can shut down Python for now
r_1_bs.to_csv('bs_coeffs_test.t')
    
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

# alternative_resids = get_residuals('alternative',X,Y,coeffs,'CAPM')['resids'] 
alternative_resids = get_residuals('alternative',X,Y,coeffs,'FF3')['resids'] 
rss_unrestricted = sum([a**2 for a in alternative_resids.values])  

tau_hat = (rss_restricted - rss_unrestricted)/rss_unrestricted

B = 5 # number of bootstrap replications
taus = [0]*B
# alternative_resids_centered = get_residuals('alternative',X,Y,coeffs,'CAPM')['centered_resids'] 
alternative_resids_centered = get_residuals('alternative',X,Y,coeffs,'FF3')['centered_resids']

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
    # bs_alt_est_coeffs.columns = ['const','rm_rf','r_1']
    bs_alt_est_coeffs.columns = ['const','rm_rf','smb','hml','r_1']
    # bs_alt_resids = get_residuals('alternative',X_recur,Y_recur,bs_alt_est_coeffs,'CAPM')['resids']
    bs_alt_resids = get_residuals('alternative',X_recur,Y_recur,bs_alt_est_coeffs,'FF3')['resids']
    bs_alt_rss = sum([a**2 for a in bs_alt_resids.values])  
    # it's not entirely clear based on sources what one is meant to do here:
    # it seems one would calculate null RSS in the BS sample, and use this for test statistic
    bs_null_resids = get_residuals('null',X_recur,Y_recur)['resids']
    bs_null_rss = sum([a**2 for a in bs_null_resids.values])
    taus[i] = (bs_null_rss - bs_alt_rss)/bs_alt_rss # T* in null distribution

###############################################################################

### old parallel code ###
    
# run for loop in parallel
# with 8 cores and about 10 years of data, this runs in ~ 25 secs
# using solve() instead of inv() here shaves 2-3 secs
# results = Parallel(n_jobs=numcores)(delayed(wls)(X=X,Y=Y,target=date,wt_fxn='gaussian',h=h) for date in datelist)

