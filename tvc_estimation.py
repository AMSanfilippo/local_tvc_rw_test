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
model = 'CAPM' # 'FF3'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,model)
# test FF3 specification, assuming lag-one AR in returns
# X = gen_X(data,asset,'FF3')
Y = (np.matrix(data.loc[1:,asset].values).T)

# iterate through normalized dates
normalized_datelist = np.divide(range(len(Y)),len(Y))

###############################################################################

# optimal bandwidth procedure
# this will take a while: about 8.5 mins for 10 bandwidths
# note: assume normalized_datelist defined as above
# normalized_datelist = np.divide(range(len(Y)),len(Y))

test_bandwidths = [0.1,0.2] # list of bandwidths to test; not fully filled 

X_cp = X.copy()
Y_cp = Y.copy()
W_loocv = np.zeros(((len(Y)-1),(len(Y)-1)),float)

cv_mses = []

for h in test_bandwidths:
    print(h)
    # compute and store kernel weights for this bandwidth
    # the most time-consuming part of the procedure
    wtmat_h = np.matrix([0]*len(normalized_datelist))
    for tau in normalized_datelist:
        outwts = np.asmatrix(gauss(tau,normalized_datelist,h))
        wtmat_h = np.concatenate((wtmat_h,outwts))

    wtmat_h = wtmat_h[1:,:] # ith row = wts. for estimating ith set of coeffs.
    
    sse = 0 # track sum of sq. errors
    
    # loocv
    for d in range(len(Y_cp)):
        X_loocv = np.delete(X_cp,d,axis=0) # this does NOT change X_cp
        Y_loocv = np.delete(Y_cp,d,axis=0) # this does NOT change Y_cp
        wts_loocv = np.delete(wtmat_h[d],d) # this does NOT change wtmat
        np.fill_diagonal(W_loocv,wts_loocv)
        out_loocv = wls(X_loocv,Y_loocv,W_loocv) # coefficient estimates for time t, using all times s != t
        sq_error = np.power(np.subtract(Y[d],np.dot(X_cp[d],out_loocv.T)),2)
        sse = np.add(sse,sq_error)
    
    mse = np.array(np.divide(sse,len(Y_cp)))[0] # loocv MSE with bandwidth = h
    cv_mses.append(mse[0]) 

results = pd.Series(cv_mses,index=test_bandwidths)

###############################################################################

h = 0.1 # bandwidth; in practice, should use optimal value found above

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
if model == 'CAPM':
    coeffs.columns = ['const','rm_rf','r_1']
elif model == 'FF3':
    coeffs.columns = ['const','rm_rf','smb','hml','r_1']
    
coeffs.to_csv('output/nonparam_coeff_ests_' + asset + '_' + model + '.csv')

# plot fitted values
fitted = get_residuals('alternative',X,Y,coeffs,model)['fitted']

x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y, color='r',label='returns')
line2, = ax.plot(x, fitted.values,color='b',dashes=[5,5],label='fitted values')
ax.legend(loc='lower left')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Returns and fitted values, ' + asset + ', ' + model)

# plt.show()
plt.savefig('figures/fitted_' + asset + '_' + model + '.jpg')

###############################################################################

# NOTE: use same normalized_datelist, h, and wtmat/W as in the above estimation

centered_resids = get_residuals('alternative',X,Y,coeffs,model)['centered_resids'] 
   
r_0 = X[0,-1] # lagged return value to serve as "seed" for DGP
X_cp = X.copy() # X matrix for DGP

B = 100 # number of bootstrap replications
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

r_1_bs.to_csv('output/bs_coeff_ests_' + asset + '_' + model + '.csv')
    
# earlier: had saved bootstrap output FFR. reload now
r_1_bs = pd.read_csv('output/bs_coeffs_test.csv')

r_1_hat = coeffs['r_1'] # point estimates for lag-one coefficient
bs_sd = np.std(r_1_bs,axis=1,ddof=1)
Q = (r_1_bs.sub(r_1_hat,axis=0)).divide(bs_sd,axis=0) # Q = (r_1_hat - r_1*)/sd(r_1*) for all t

# empirical critical value for 95% pointwise CI
c = np.percentile(Q,97.5,axis=1,interpolation='higher') # may want to think about different interpolation param
me =  np.multiply(bs_sd.values,c) # margin of error

# CIs: b_hat +/- c*sd for each t
CI_lb = np.subtract(r_1_hat.values,me) 
CI_ub = np.add(r_1_hat.values,me)

CI_df = pd.DataFrame({'lb':CI_lb,'pt_est':r_1_hat.values,'ub':CI_ub})
CI_df.to_csv('output/bs_coeff_CIs_' + asset + '_' + model + '.csv')

# plot 95% pointwise CIs
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, CI_lb, dashes = [5,5], linewidth=2,color='b',label='confidence bound')
line2, = ax.plot(x, CI_ub, dashes = [5,5], linewidth=2,color='b')
line3, = ax.plot(x, r_1_hat.values, linewidth=2, color='r',label='point estimate')
line4, = ax.plot(x,[0]*len(x),dashes = [7,3],color='grey')
ax.legend(loc='lower left')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('95% pointwise confidence intervals for coefficient on single-day lagged returns, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/pointwiseCIs_' + asset + '_' + model + '.jpg')

###############################################################################

ols_out = ols(X,Y) 

null_resids = get_residuals('null',X,Y)['resids']

rss_restricted = sum(np.power(null_resids.values,2))

alternative_resids = get_residuals('alternative',X,Y,coeffs,model)['resids'] 
rss_unrestricted = sum(np.power(alternative_resids.values,2))  

tau_hat = (rss_restricted - rss_unrestricted)/rss_unrestricted # test statistic to compare to null distribution

B = 100 # number of bootstrap replications
taus = [0]*B
alternative_resids_centered = get_residuals('alternative',X,Y,coeffs,model)['centered_resids'] 

r_0 = X[0,-1] # lagged return value to serve as "seed" for DGP
X_cp = X.copy() # X matrix for DGP

# compute null distribution of the test statistic using bootstrap
for i in range(B):
    print('bs iteration: ', i)
    bs_resids = bs_resample(alternative_resids_centered.values) 
    recursive = gen_recursive('null',X_cp,r_0,bs_resids,ols_out)
    X_cp[:,-1] = recursive[0]
    Y_recur = recursive[1]
    results = np.matrix([0]*len(X.T))
    # estimate nonparametric regression on bootstrap dataset, generated under null DGP
    for d in range(len(Y_recur)):
        np.fill_diagonal(W,wtmat[d]) # diagonal matrix of weights
        out = wls(X_cp,Y_recur,W)
        # return coefficient estimates and residuals
        results = np.concatenate((results,out))
    bs_alt_est_coeffs = pd.DataFrame(results[1:,:])
    if model == 'CAPM':
        bs_alt_est_coeffs.columns = ['const','rm_rf','r_1']
    elif model == 'FF3':
        bs_alt_est_coeffs.columns = ['const','rm_rf','smb','hml','r_1']
    bs_alt_resids = get_residuals('alternative',X_cp,Y_recur,bs_alt_est_coeffs,model)['resids']
    bs_alt_rss = sum(np.power(bs_alt_resids.values,2))  
    # calculate null model RSS in bootstrap dataset
    bs_null_resids = get_residuals('null',X_cp,Y_recur)['resids']
    bs_null_rss = sum(np.power(bs_null_resids.values,2))
    taus[i] = (bs_null_rss - bs_alt_rss)/bs_alt_rss # T* in null distribution

f = open('output/taus_' + asset + '_' + model + '.txt','w')
for t in taus:
    f.write(str(t)+',')
f.close()

sum(taus > tau_hat) # 0. reject H0.

###############################################################################






