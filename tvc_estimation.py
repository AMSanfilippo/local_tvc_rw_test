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

asset = 'BusEq'
model = 'CAPM'

X = gen_X(data,asset,model)
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

r_1_bs.to_csv('output/' + asset + '/bs_coeff_ests_' + model + '.csv')
    

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
CI_df.to_csv('output/' + asset + '/bs_coeff_CIs_' + model + '.csv')

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
plt.savefig('figures/' + asset + '/pointwiseCIs_' + model + '.jpg')

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

# generate code to shade plot where phi is significant
def gen_axvspan(inds):
    for i in range(len(inds) - 1):
        start = inds[i]
        end = inds[i+1]
        if end - start == 1:
            print('ax.axvspan(x.loc[' + str(start) + '],x.loc[' + str(end) + '], facecolor=\'grey\', alpha = 0.25)')
  
def sumstats_ofint(inds,rets,sig_lev):
    start_ends = []
    stdevs = []
    d1s = []
    d2s = []
    block_start = inds[0]
    block_end = inds[0]
    for i in range(len(inds) - 1):
        start = inds[i]
        end = inds[i+1]
        if end - start == 1:
            block_end = end
            if (i+1) == (len(inds) - 1): # if last entry the list of indices
                start_ends.append([block_start,block_end])
                rets_ofint = rets[block_start:block_end+1]
                stdevs.append(np.std(rets_ofint,ddof=1))
                d1 = np.subtract(rets_ofint[1:],rets_ofint[:-1])
                d1s.append(np.mean(d1))
                if len(d1) > 1:
                    d2 = np.subtract(d1[1:],d1[:-1])
                    d2s.append(np.mean(d2))
        else:
            start_ends.append([block_start,block_end])
            rets_ofint = rets[block_start:block_end+1]
            if len(rets_ofint) > 1:
                stdevs.append(np.std(rets_ofint,ddof=1))
                d1 = np.subtract(rets_ofint[1:],rets_ofint[:-1])
                d1s.append(np.mean(d1))
                if len(d1) > 1:
                    d2 = np.subtract(d1[1:],d1[:-1])
                    d2s.append(np.mean(d2))
            block_start = end
            block_end = end
    for i in range(len(start_ends)):
        if sig_lev == 's':
            print('ax.hlines(' + str(stdevs[i]) + ', ' + str((start_ends[i][0])/len(rets)) + ', ' + str((start_ends[i][1])/len(rets)) + ', color=\'g\')')
        elif sig_lev == 'ns':
            print('ax.hlines(' + str(stdevs[i]) + ', ' + str((start_ends[i][0])/len(rets)) + ', ' + str((start_ends[i][1])/len(rets)) + ', color=\'b\')')
    return [start_ends, stdevs, d1s, d2s]
        
      
###############################################################################

# exploratory analysis: durbl

asset = 'Durbl'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)

durbl_CIs = pd.read_csv('output/Durbl/bs_coeff_CIs_' + model + '.csv',index_col=0)

# date range where phi is significantly < 0
inds = list(durbl_CIs[durbl_CIs.ub < 0].index)

# standard deviation of daily return during significant time periods
sumstats_ofint(inds,Y,'s')

nonsig_inds = list(durbl_CIs[durbl_CIs.ub >= 0].index)
sumstats_ofint(nonsig_inds,Y,'ns') 

# plot summary stats
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))
ax.hlines(1.56458988072, 0.582042113627, 0.735399284863, color='g')
ax.hlines(1.11435130276, 0.0, 0.572904251093, color='b')
ax.hlines(1.59617554604, 0.789034564958, 0.999602701629, color='b')
ax.set_title('Average daily standard deviation of returns, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/daily_stdev' + model + '.jpg')

gen_axvspan(inds)

# plot returns in this time period
x = data.loc[inds[0]:inds[-1],'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y[inds[0]:(inds[-1]+1)], color='r',label='excess returns')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns where phi < 0, Durbl, ' + model)

#plt.show()
plt.savefig('figures/Durbl/returns_ofint_' + model + '.jpg')

subsect_Y = Y[inds[0]:(inds[-1]+1)].copy()
cumulative_rets = np.cumprod(np.add(np.divide(subsect_Y,100),1)).tolist()[0]

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, cumulative_rets, color='r',label='cumulative return')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns where phi < 0, Durbl' + model)

#plt.show()
plt.savefig('figures/Durbl/cumrets_ofint_' + model + '.jpg')

# full-period cumulative returns with shaded area where phi > 0
full_cumulative_rets = np.cumprod(np.add(np.divide(Y.copy(),100),1)).tolist()[0]
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, full_cumulative_rets, color='r',label='cumulative return')
# axvspan code goes here
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns, Durbl, ' + model)

#plt.show()
plt.savefig('figures/Durbl/full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# exploratory analysis: telcm

asset = 'Telcm'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
telcm_CIs = pd.read_csv('output/Telcm/bs_coeff_CIs_' + model + '.csv',index_col=0)

inds = list(telcm_CIs[telcm_CIs.lb > 0].index) 

# standard deviation of daily return during significant time periods
sumstats_ofint(inds,Y,'s')

nonsig_inds = list(telcm_CIs[telcm_CIs.lb <= 0].index)
sumstats_ofint(nonsig_inds,Y,'ns') 

# plot summary stats
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))
ax.hlines(0.785164629226, 0.0305919745729, 0.0528406833532, color='g')
ax.hlines(1.23763764535, 0.297576479936, 0.303933253874, color='g')
ax.hlines(2.1099890275, 0.671831545491, 0.717520858164, color='g')
ax.hlines(2.84657436874, 0.858164481526, 0.874851013111, color='g')
ax.hlines(0.944258968716, 0.978943186333, 0.980135081446, color='g')
ax.hlines(0.841642943186, 0.0, 0.0301946762018, color='b')
ax.hlines(0.783344108161, 0.0532379817243, 0.297179181565, color='b')
ax.hlines(1.38795854289, 0.304330552245, 0.67143424712, color='b')
ax.hlines(1.73838179283, 0.717918156536, 0.857767183155, color='b')
ax.hlines(1.80283724944, 0.875248311482, 0.978545887962, color='b')
ax.hlines(0.682743907477, 0.980532379817, 0.999602701629, color='b')
ax.set_title('Average daily standard deviation of returns, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/daily_stdev' + model + '.jpg')

gen_axvspan(inds)

x = data.loc[start_ind:end_ind,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y[start_ind:end_ind+1], color='r',label='excess returns')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns where phi > 0, Telcm, ' + model)
#plt.show()
plt.savefig('figures/Telcm/returns_ofint_' + model + '.jpg')

subsect_Y = Y[start_ind:end_ind+1].copy()
cumulative_rets = np.cumprod(np.add(np.divide(subsect_Y,100),1)).tolist()[0]

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, cumulative_rets, color='r',label='cumulative return')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns where phi > 0, Telcm, ' + model)

#plt.show()
plt.savefig('figures/Telcm/cumrets_ofint_' + model + '.jpg')

# full-period cumulative returns with shaded area where phi > 0
full_cumulative_rets = np.cumprod(np.add(np.divide(Y.copy(),100),1)).tolist()[0]
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, full_cumulative_rets, color='r',label='cumulative return')
# axvspan code goes here
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns, Telcm, ' + model)

#plt.show()
plt.savefig('figures/Telcm/full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# exploratory analysis: buseq

asset = 'BusEq'
model = 'CAPM'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
buseq_CIs = pd.read_csv('output/BusEq/bs_coeff_CIs_' + model + '.csv',index_col=0)

inds = list(buseq_CIs[buseq_CIs.lb < 0].index)

# standard deviation of daily return during significant time periods
sumstats_ofint(inds,Y,'s')

nonsig_inds = list(buseq_CIs[buseq_CIs.ub >= 0].index)
sumstats_ofint(nonsig_inds,Y,'ns') 

# plot summary stats
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))
ax.hlines(1.54295774804, 0.162892332141, 0.182359952324, color='g')
ax.hlines(1.45151175065, 0.249503377036, 0.297576479936, color='g')
ax.hlines(2.12862136189, 0.437028208184, 0.502185141041, color='g')
ax.hlines(1.18932403771, 0.559793404847, 0.56098529996, color='g')
ax.hlines(3.50923517026, 0.618593563766, 0.737385776718, color='g')
ax.hlines(2.40444264736, 0.769169646404, 0.772348033373, color='g')
ax.hlines(2.62312926032, 0.795788637267, 0.891140246325, color='g')
ax.hlines(1.0684126631, 0.0, 0.16249503377, color='b')
ax.hlines(1.48560048746, 0.182757250695, 0.249106078665, color='b')
ax.hlines(1.55080634419, 0.297973778308, 0.436630909813, color='b')
ax.hlines(2.0880660948, 0.502582439412, 0.559396106476, color='b')
ax.hlines(1.88293210167, 0.561382598331, 0.618196265395, color='b')
ax.hlines(2.40131871559, 0.737783075089, 0.768772348033, color='b')
ax.hlines(2.51419912819, 0.772745331744, 0.795391338896, color='b')
ax.hlines(1.61850538379, 0.891537544696, 0.999602701629, color='b')
ax.set_title('Average daily standard deviation of returns, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/daily_stdev' + model + '.jpg')

gen_axvspan(inds[:300])

x = data.loc[start_ind:end_ind,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y[start_ind:end_ind+1], color='r',label='excess returns')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns where phi < 0, BusEq, ' + model)
#plt.show()
plt.savefig('figures/BusEq/returns_ofint_' + model + '.jpg')

# plot cumulative returns in period of interest
subsect_Y = Y[start_ind:end_ind+1].copy()
cumulative_rets = np.cumprod(np.add(np.divide(subsect_Y,100),1)).tolist()[0]

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, cumulative_rets, color='r',label='cumulative return')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns where phi < 0, BusEq, ' + model)

#plt.show()
plt.savefig('figures/BusEq/cumrets_ofint_' + model + '.jpg')

# full-period cumulative returns with shaded area where phi < 0
full_cumulative_rets = np.cumprod(np.add(np.divide(Y.copy(),100),1)).tolist()[0]
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, full_cumulative_rets, color='r',label='cumulative return')
# axvspan code goes here
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns, BusEq, ' + model)

#plt.show()
plt.savefig('figures/BusEq/full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# exploratory analysis: manuf

asset = 'Manuf'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
manuf_CIs = pd.read_csv('output/Manuf/bs_coeff_CIs_' + model + '.csv',index_col=0)

inds = list(manuf_CIs[manuf_CIs.lb > 0].index)

# standard deviation of daily return during significant time periods
sumstats_ofint(inds,Y,'s')

nonsig_inds = list(manuf_CIs[manuf_CIs.lb <= 0].index)
sumstats_ofint(nonsig_inds,Y,'ns') 

# plot summary stats
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))
ax.hlines(0.661417328002, 0.0, 0.0961462058006, color='g')
ax.hlines(0.815149870987, 0.342471195868, 0.353198251887, color='g')
ax.hlines(1.7835022272, 0.455303933254, 0.48708780294, color='g')
ax.hlines(0.791487108636, 0.541914978149, 0.561779896702, color='g')
ax.hlines(1.32508279438, 0.642828764402, 0.709972189114, color='g')
ax.hlines(1.37521513142, 0.728247914184, 0.828764402066, color='g')
ax.hlines(0.892282484433, 0.957091775924, 0.986094557012, color='g')
ax.hlines(0.673046840344, 0.0965435041716, 0.342073897497, color='b')
ax.hlines(1.01377470297, 0.353595550258, 0.454906634883, color='b')
ax.hlines(1.0670885801, 0.487485101311, 0.541517679778, color='b')
ax.hlines(1.45194482504, 0.562177195074, 0.642431466031, color='b')
ax.hlines(1.59310550641, 0.710369487485, 0.727850615812, color='b')
ax.hlines(1.51827508929, 0.829161700437, 0.956694477553, color='b')
ax.hlines(0.8143215082, 0.986491855383, 0.999602701629, color='b')
ax.set_title('Average daily standard deviation of returns, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/daily_stdev' + model + '.jpg')

gen_axvspan(inds[:300])

x = data.loc[start_ind:end_ind,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y[start_ind:end_ind+1], color='r',label='excess returns')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns where phi > 0, Manuf, ' + model)
#plt.show()
plt.savefig('figures/Manuf/returns_ofint_' + model + '.jpg')

# plot cumulative returns in period of interest
subsect_Y = Y[start_ind:end_ind+1].copy()
cumulative_rets = np.cumprod(np.add(np.divide(subsect_Y,100),1)).tolist()[0]

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, cumulative_rets, color='r',label='cumulative return')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns where phi > 0, Manuf, ' + model)

#plt.show()
plt.savefig('figures/Manuf/cumrets_ofint_' + model + '.jpg')

# full-period cumulative returns with shaded area where phi > 0
full_cumulative_rets = np.cumprod(np.add(np.divide(Y.copy(),100),1)).tolist()[0]
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, full_cumulative_rets, color='r',label='cumulative return')
# axvspan code goes here
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns, Manuf, ' + model)

#plt.show()
plt.savefig('figures/Manuf/full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# exploratory analysis: other

asset = 'Other'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
other_CIs = pd.read_csv('output/Other/bs_coeff_CIs_' + model + '.csv',index_col=0)

inds = list(other_CIs[other_CIs.lb > 0].index)

# standard deviation of daily return during significant time periods
sumstats_ofint(inds,Y,'s')

nonsig_inds = list(other_CIs[other_CIs.lb <= 0].index)
sumstats_ofint(nonsig_inds,Y,'ns') 

# plot summary stats
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))
ax.hlines(0.549845027418, 0.0472785061581, 0.192689709972, color='g')
ax.hlines(1.42920787203, 0.497417560588, 0.806912991657, color='g')
ax.hlines(0.747762798785, 0.0, 0.046881207787, color='b')
ax.hlines(0.976350931853, 0.193087008343, 0.497020262217, color='b')
ax.hlines(1.58429359653, 0.807310290028, 0.999602701629, color='b')
ax.set_title('Average daily standard deviation of returns, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/daily_stdev' + model + '.jpg')

gen_axvspan(inds[:300])

x = data.loc[start_ind:end_ind,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y[start_ind:end_ind+1], color='r',label='excess returns')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns where phi > 0, Other, ' + model)
#plt.show()
plt.savefig('figures/Other/returns_ofint_' + model + '.jpg')

# plot cumulative returns in period of interest
subsect_Y = Y[start_ind:end_ind+1].copy()
cumulative_rets = np.cumprod(np.add(np.divide(subsect_Y,100),1)).tolist()[0]

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, cumulative_rets, color='r',label='cumulative return')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns where phi > 0, Other, ' + model)

#plt.show()
plt.savefig('figures/Other/cumrets_ofint_' + model + '.jpg')

# full-period cumulative returns with shaded area where phi > 0
full_cumulative_rets = np.cumprod(np.add(np.divide(Y.copy(),100),1)).tolist()[0]
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, full_cumulative_rets, color='r',label='cumulative return')
# axvspan code goes here
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns, Other, ' + model)

#plt.show()
plt.savefig('figures/Other/full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# plot confidence intervals for symmetric and backward-facing estimation together
asset = 'Other'
model = 'FF5'
sym_CIs = pd.read_csv('output/' + asset + '/bs_coeff_CIs_' + model + '.csv',index_col=0)
bwf_CIs = pd.read_csv('output/' + asset + '/bwunif_bs_coeff_CIs_' + model + '.csv',index_col=0)

# plot 95% pointwise CIs, layered
x = data.loc[64:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, sym_CIs.loc[63:,'lb'].values, dashes = [5,5], linewidth=1,color='b',label='confidence bound, symmetric kernel',alpha=0.33)
line2, = ax.plot(x, sym_CIs.loc[63:,'ub'].values, dashes = [5,5], linewidth=1,color='b',alpha=0.33)
line3, = ax.plot(x, sym_CIs.loc[63:,'pt_est'].values, linewidth=2, color='b',label='point estimate, symmetric kernel')
line4, = ax.plot(x, bwf_CIs['lb'].values, dashes = [5,5], linewidth=1,color='r',label='confidence bound, backward kernel',alpha=0.33)
line5, = ax.plot(x, bwf_CIs['ub'].values, dashes = [5,5], linewidth=1,color='r',alpha=0.33)
line6, = ax.plot(x, bwf_CIs['pt_est'].values, linewidth=2, color='r',label='point estimate, backward kernel')
line7, = ax.plot(x,[0]*len(x),dashes = [7,3],color='grey')
ax.legend(loc='lower left')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Comparison of pointwise confidence intervals for coefficient on single-day lagged returns, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/pointwiseCIs_layered_' + model + '.jpg')

###############################################################################



