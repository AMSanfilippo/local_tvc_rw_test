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

###############################################################################

asset = 'Hlth '
model = 'FF5'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,model)
# test FF3 specification, assuming lag-one AR in returns
# X = gen_X(data,asset,'FF3')
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T) # excess returns

# iterate through normalized dates
normalized_datelist = np.divide(range(len(Y)),len(Y))

###############################################################################

# optimal bandwidth procedure
# note: assume normalized_datelist defined as above
# normalized_datelist = np.divide(range(len(Y)),len(Y))

test_bandwidths = list(np.divide(range(1,26),500)) # list of bandwidths to test
# (tbh the optimal bandwidth will never be above 0.05 so why bother checking.)

X_cp = X.copy()
Y_cp = Y.copy()
W_loocv = np.zeros(((len(Y)-1),(len(Y)-1)),float)

cv_mses = []

for h in test_bandwidths:
    print(h)
    
    # compute and store kernel weights for this bandwidth
    # not using concatenate; v faster much better
    wtmat_h = np.matrix(np.zeros((len(normalized_datelist),len(normalized_datelist))))
    ind = 0
    for tau in normalized_datelist:
        outwts = gauss(tau,normalized_datelist,h)
        wtmat_h[ind,:] = outwts
        ind += 1
    
    sse = 0 # track sum of sq. errors
    
    # loocv
    for d in range(len(Y_cp)):
        X_loocv = np.delete(X_cp,d,axis=0) # this does NOT change X_cp
        Y_loocv = np.delete(Y_cp,d,axis=0) # this does NOT change Y_cp
        wts_loocv = np.delete(wtmat_h[d],d) # this does NOT change wtmat
        np.fill_diagonal(W_loocv,wts_loocv)
        out_loocv = wls(X_loocv,Y_loocv,W_loocv) # coefficient estimates for time t, using all times s != t
        sq_error = np.power(np.subtract(Y_cp[d],np.dot(X_cp[d],out_loocv.T)),2)
        sse = np.add(sse,sq_error)
    
    mse = np.array(np.divide(sse,len(Y_cp)))[0] # loocv MSE with bandwidth = h
    cv_mses.append(mse[0]) 

results = pd.Series(cv_mses,index=test_bandwidths)
results.to_csv('output/' + asset + '/loocv_' + model + '.csv')

h = results[results==min(results)].index[0] # optimal bandwidth

###############################################################################

# NOTE: all of the below assumes the use of the optimal bandwidth from above

# to speed up regression: compute and store kernel weights first
# note: weights are f(h,n) only
# a given bootstrap procedure will always use the same h and n, so this is valid
wtmat = np.matrix(np.zeros((len(normalized_datelist),len(normalized_datelist))))
ind = 0
for tau in normalized_datelist:
    outwts = gauss(tau,normalized_datelist,h)
    wtmat[ind,:] = outwts
    ind += 1

W = np.zeros((len(Y),len(Y)),float) # empty, to become diagonal weight matrix
results = np.asmatrix(np.zeros((len(X),len(X.T))))

# estimate nonparametric regression
for d in range(len(Y)):
    np.fill_diagonal(W,wtmat[d]) # diagonal matrix of weights
    out = wls(X,Y,W)
    # return coefficient estimates and residuals
    results[d,:] = out

coeffs = pd.DataFrame(results)
if model == 'CAPM':
    coeffs.columns = ['const','rm_rf','r_1']
elif model == 'FF3':
    coeffs.columns = ['const','rm_rf','smb','hml','r_1']
elif model == 'FF5':
    coeffs.columns = ['const','rm_rf','smb','hml','rmw','cma','r_1']
    
coeffs.to_csv('output/' + asset + '/nonparam_coeff_ests_' + model + '.csv')

# ts plot of fitted values
fitted = get_residuals('alternative',X,Y,coeffs,model)['fitted']

x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y, color='r',label='excess returns')
line2, = ax.plot(x, fitted.values,color='b',dashes=[5,5],label='fitted values')
ax.legend(loc='lower left')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns and fitted values, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/fitted_' + model + '.jpg')

# histogram of residuals
resids = get_residuals('alternative',X,Y,coeffs,model)['resids']

fig, ax = plt.subplots(figsize=(7,5))
ax.hist(resids.values,bins=100)
ax.set_title('Histogram of residuals, ' + asset + ', ' + model)
#plt.show()
plt.savefig('figures/' + asset + '/resid_hist_' + model + '.jpg')

# ts plot of residuals
fig, ax = plt.subplots(figsize=(10,5))
line1, = ax.plot(x, resids.values, color='b')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Residuals, ' + asset + ', ' + model)
#plt.show()
plt.savefig('figures/' + asset + '/resid_ts_' + model + '.jpg')

# ts plot of coefficient point estimates
fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, coeffs['const'].values, color='r',label='constant (alpha)')
line2, = ax.plot(x, coeffs['rm_rf'].values, color='g',label='market return')
line3, = ax.plot(x, coeffs['r_1'].values, color='b',label='lag-one return')
if model == 'FF3':
    line4, = ax.plot(x, coeffs['smb'].values, color='y',label='size factor')
    line5, = ax.plot(x, coeffs['hml'].values, color='c',label='value factor')
if model == 'FF5':
    line4, = ax.plot(x, coeffs['smb'].values, color='y',label='size factor')
    line5, = ax.plot(x, coeffs['hml'].values, color='c',label='value factor')
    line6, = ax.plot(x, coeffs['rmw'].values, color='orange',label='profitability factor')
    line7, = ax.plot(x, coeffs['cma'].values, color='purple',label='investment factor')
line8, = ax.plot(x,[0]*len(x),dashes = [7,3],color='grey')
    
ax.legend(loc='lower left')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Coefficient point estimates, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/coeffs_' + model + '.jpg')

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
    results = np.asmatrix(np.zeros((len(X_cp),len(X_cp.T))))
    
    # estimate nonparametric regression on bootstrap dataset
    for d in range(len(Y_recur)):
        np.fill_diagonal(W,wtmat[d])
        out = wls(X_cp,Y_recur,W)
        results[d,:] = out
        
    r_1_bs[i] = results[:,-1]

r_1_bs.to_csv('output/' + assets + '/bs_coeff_ests_' + model + '.csv')
    

r_1_hat = coeffs['r_1'] # point estimates for lag-one coefficient
bs_sd = np.std(r_1_bs,axis=1,ddof=1)
Q = (r_1_bs.sub(r_1_hat,axis=0)).divide(bs_sd,axis=0) # Q = (r_1_hat - r_1*)/sd(r_1*) for all t

# empirical critical value for 95% pointwise CI
c = np.percentile(Q,97.5,axis=1,interpolation='midpoint') 
me =  np.multiply(bs_sd.values,c) # margin of error

# CIs: b_hat +/- c*sd for each t
CI_lb = np.subtract(r_1_hat.values,me) 
CI_ub = np.add(r_1_hat.values,me)

CI_df = pd.DataFrame({'lb':CI_lb,'pt_est':r_1_hat.values,'ub':CI_ub})
CI_df.to_csv('output/' + assets + '/bs_coeff_CIs_' + model + '.csv')

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
plt.savefig('figures/' + assets + '/pointwiseCIs_' + model + '.jpg')

###############################################################################

# specification test 
# H0: all coefficients time-invariate vs.
# HA: at least one coefficient time-varying

ols_out = ols(X,Y) 

ols_fitted = get_residuals('null',X,Y)['fitted']

# plot ols fitted values
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y, color='r',label='excess returns')
line2, = ax.plot(x, ols_fitted.values,color='b',dashes=[5,5],label='ols fitted values')
ax.legend(loc='lower left')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns and OLS fitted values, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/fitted_ols_' + model + '.jpg')

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
    
    results = np.asmatrix(np.zeros((len(X_cp),len(X_cp.T))))
    # estimate nonparametric regression on bootstrap dataset, generated under null DGP
    for d in range(len(Y_recur)):
        np.fill_diagonal(W,wtmat[d])
        out = wls(X_cp,Y_recur,W)
        results[d,:] = out
    
    bs_alt_est_coeffs = pd.DataFrame(results)
    if model == 'CAPM':
        bs_alt_est_coeffs.columns = ['const','rm_rf','r_1']
    elif model == 'FF3':
        bs_alt_est_coeffs.columns = ['const','rm_rf','smb','hml','r_1']
    elif model == 'FF5':
        bs_alt_est_coeffs.columns = ['const','rm_rf','smb','hml','rmw','cma','r_1']
        
    bs_alt_resids = get_residuals('alternative',X_cp,Y_recur,bs_alt_est_coeffs,model)['resids']
    bs_alt_rss = sum(np.power(bs_alt_resids.values,2))  
    # calculate null model RSS in bootstrap dataset
    bs_null_resids = get_residuals('null',X_cp,Y_recur)['resids']
    bs_null_rss = sum(np.power(bs_null_resids.values,2))
    taus[i] = (bs_null_rss - bs_alt_rss)/bs_alt_rss # T* in null distribution

f = open('output/' + asset + '/taus_' + model + '.txt','w')
for t in taus:
    f.write(str(t)+',')
f.close()

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(taus,bins=15) 
ax.set_title('Distribution of tau statistic under H0: all coefficients time-invariant; ' + asset + ', ' + model)
ax.axvline(tau_hat, color='r', linestyle='dashed', linewidth=2,label='test statistic value')
ax.legend(loc='upper left')

#plt.show()
plt.savefig('figures/' + asset + '/tau_hist_01_' + model +'.jpg')

sum(taus > tau_hat) # 0. reject H0.

###############################################################################

# specification test
# H0: (all coefficients time-varying and) coefficient on lagged return = 0 for all t
# HA: (all coefficients time-varying and) coefficient on lagged return != 0 for some t

X_cp_nolag = np.delete(X.copy(),-1,1)
results_nolag = np.asmatrix(np.zeros((len(X_cp_nolag),len(X_cp_nolag.T))))

# estimate nonparametric regression on null model
for d in range(len(Y)):
    np.fill_diagonal(W,wtmat[d]) # diagonal matrix of weights
    out = wls(X_cp_nolag,Y,W)
    # return coefficient estimates and residuals
    results_nolag[d,:] = out

coeffs_nolag = pd.DataFrame(results_nolag)
if model == 'CAPM':
    coeffs_nolag.columns = ['const','rm_rf']
elif model == 'FF3':
    coeffs_nolag.columns = ['const','rm_rf','smb','hml']
elif model == 'FF5':
    coeffs_nolag.columns = ['const','rm_rf','smb','hml','rmw','cma']
    
coeffs_nolag.to_csv('output/' + asset + '/nonparam_null_ests_' + model + '.csv')

null_resids = get_residuals('alternative',X_cp_nolag,Y,coeffs_nolag,model)['resids']
rss_restricted = sum(np.power(null_resids.values,2))  

# using results from above estimation of alternative model
alternative_resids = get_residuals('alternative',X,Y,coeffs,model)['resids'] 
rss_unrestricted = sum(np.power(alternative_resids.values,2))  

tau_hat = (rss_restricted - rss_unrestricted)/rss_unrestricted # test statistic to compare to null distribution


coeffs_nolag['r_1'] = np.zeros(len(coeffs_nolag)) # under null, beta(r_1) = 0 for all t
r_0 = X[0,-1] # lagged return value to serve as "seed" for DGP

# matrix for specification testing: beta(r_1) = 0 for all t vs. beta(r_1) != 0 for some t 
X_cp_spec = np.concatenate((X_cp_nolag,np.matrix([0]*len(X_cp_nolag)).T),axis=1) 

B = 100 # number of bootstrap replications
taus = [0]*B
alternative_resids_centered = get_residuals('alternative',X,Y,coeffs,model)['centered_resids'] 

# compute null distribution of the test statistic using bootstrap
for i in range(B):
    print('bs iteration: ', i)
    bs_resids = bs_resample(alternative_resids_centered.values) 
    recursive = gen_recursive('alternative',X_cp_nolag,r_0,bs_resids,coeffs_nolag)
    X_cp_spec[:,-1] = recursive[0]
    Y_recur = recursive[1]
    results = np.asmatrix(np.zeros((len(X),len(X.T))))
    results_null = np.asmatrix(np.zeros((len(X_cp_nolag),len(X_cp_nolag.T))))
    
    # simultaneously: 
    # estimate nonparametric alternative model on bootstrap dataset, generated under null DGP
    # estimate nonparametric null model on bootstrap dataset
    for d in range(len(Y_recur)):
        np.fill_diagonal(W,wtmat[d])
        out = wls(X_cp_spec,Y_recur,W)
        out_null = wls(X_cp_nolag,Y_recur,W)
        results[d,:] = out
        results_null[d,:] = out_null
        
    bs_alt_est_coeffs = pd.DataFrame(results)
    bs_null_est_coeffs = pd.DataFrame(results_null)
    if model == 'CAPM':
        bs_alt_est_coeffs.columns = ['const','rm_rf','r_1']
        bs_null_est_coeffs.columns = ['const','rm_rf']
    elif model == 'FF3':
        bs_alt_est_coeffs.columns = ['const','rm_rf','smb','hml','r_1']
        bs_null_est_coeffs.columns = ['const','rm_rf','smb','hml']
    elif model == 'FF5':
        bs_alt_est_coeffs.columns = ['const','rm_rf','smb','hml','rmw','cma','r_1']
        bs_null_est_coeffs.columns = ['const','rm_rf','smb','hml','rmw','cma']
        
    bs_alt_resids = get_residuals('alternative',X_cp_spec,Y_recur,bs_alt_est_coeffs,model)['resids']
    bs_alt_rss = sum(np.power(bs_alt_resids.values,2))  
    
    bs_null_resids = get_residuals('alternative',X_cp_nolag,Y_recur,bs_null_est_coeffs,model)['resids']
    bs_null_rss = sum(np.power(bs_null_resids.values,2))  

    taus[i] = (bs_null_rss - bs_alt_rss)/bs_alt_rss # T* in null distribution


f = open('output/' + asset + '/taus_alt' + model + '.txt','w')
for t in taus:
    f.write(str(t)+',')
f.close()

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(taus,bins=15) 
ax.set_title('Distribution of tau statistic under H0: phi = 0 for all times t, ' + asset + ', ' + model)
ax.axvline(tau_hat, color='r', linestyle='dashed', linewidth=2,label='test statistic value')
ax.legend(loc='upper left')

#plt.show()
plt.savefig('figures/' + asset + '/tau_hist_02_' + model +'.jpg')

sum(taus > tau_hat) # 0. reject H0.


###############################################################################

