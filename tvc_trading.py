#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:45:38 2018

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

###############################################################################

data = clean_data('19940101','20031230')

###############################################################################

asset = 'Money'
model = 'FF5'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,model)
# test FF3 specification, assuming lag-one AR in returns
# X = gen_X(data,asset,'FF3')
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T) # excess returns

###############################################################################

window = 63 # 3-month rolling window. (tbh I don't know what to do with this.)

###############################################################################

# estimate nonparametric regression using backward-facing uniform kernel

results = np.asmatrix(np.zeros((len(X),len(X.T))))

for d in range(window,len(X)):
	X_sub = X[d-window:d,:] # only observations within most recent time window
	Y_sub = Y[d-window:d]
	out = np.linalg.solve((X_sub.T).dot(X_sub),(X_sub.T).dot(Y_sub)) # coefficient estimates for time t
	results[d,:] = out.T

coeffs = pd.DataFrame(results[window:,:])
if model == 'CAPM':
    coeffs.columns = ['const','rm_rf','r_1']
elif model == 'FF3':
    coeffs.columns = ['const','rm_rf','smb','hml','r_1']
elif model == 'FF5':
    coeffs.columns = ['const','rm_rf','smb','hml','rmw','cma','r_1']
    
coeffs.to_csv('output/' + asset + '/bwunif_nonparam_coeff_ests_' + model + '.csv')

# fitted values from the estimation, starting {window} days in
fitted = get_residuals('alternative',X[window:,:],Y[window:],coeffs,model)['fitted']

x = data.loc[window+1:,'Date']

# plot of fitted values
fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, Y[window:], color='r',label='excess returns')
line2, = ax.plot(x, fitted.values,color='b',dashes=[5,5],label='fitted values')
ax.legend(loc='lower left')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Excess returns and fitted values, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/bwunif_fitted_' + model + '.jpg')

# histogram of residuals
resids = get_residuals('alternative',X[window:,:],Y[window:],coeffs,model)['resids']

fig, ax = plt.subplots(figsize=(7,5))

ax.hist(resids.values,bins=100)
ax.set_title('Histogram of residuals, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/bwunif_resid_hist_' + model + '.jpg')

# ts plot of residuals
fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, resids.values, color='b')
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Residuals, ' + asset + ', ' + model)

#plt.show()
plt.savefig('figures/' + asset + '/bwunif_resid_ts_' + model + '.jpg')

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
plt.savefig('figures/' + asset + '/bwunif_coeffs_' + model + '.jpg')

###############################################################################

centered_resids = get_residuals('alternative',X[window:,:],Y[window:],coeffs,model)['centered_resids'] 
   
r_0 = X[window,-1] # lagged return value to serve as "seed" for DGP
X_cp = X.copy() # X matrix for DGP
Y_cp = Y.copy() # Y matrix for bootstrap estimation

B = 500 # number of bootstrap replications
r_1_bs = pd.DataFrame(np.zeros((len(Y[window:]),B),float))
for i in range(B):
    if i%20 == 0:
        print('bs replication: ', i+1)
    bs_resids = bs_resample(centered_resids.values)
    recursive = gen_recursive('alternative',X_cp[window:,:],r_0,bs_resids,coeffs,model)
    X_cp[window:,-1] = recursive[0]
    Y_cp[window:] = recursive[1]

    results = np.asmatrix(np.zeros((len(X_cp),len(X_cp.T))))
    
    # estimate nonparametric regression on bootstrap dataset
    for d in range(window,len(X)):
        X_sub = X_cp[d-window:d,:] # only observations within most recent time window
        Y_sub = Y_cp[d-window:d]
        out = np.linalg.solve((X_sub.T).dot(X_sub),(X_sub.T).dot(Y_sub)) # coefficient estimates for time t
        results[d,:] = out.T
        
    r_1_bs[i] = results[window:,-1]

r_1_bs.to_csv('output/' + asset + '/bwunif_bs_coeff_ests_' + model + '.csv')

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
CI_df.to_csv('output/' + asset + '/bwunif_bs_coeff_CIs_' + model + '.csv')

x = data.loc[window+1:,'Date']

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
plt.savefig('figures/' + asset + '/bwunif_pointwiseCIs_' + model + '.jpg')































