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

asset = 'Telcm'
model = 'FF5'

# test CAPM specification, assuming lag-one AR in returns
X = gen_X(data,asset,model)
# test FF3 specification, assuming lag-one AR in returns
# X = gen_X(data,asset,'FF3')
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T) # excess returns

###############################################################################

window = 63 # 3-month rolling window. (64 days)

###############################################################################

# estimate nonparametric regression using backward-facing uniform kernel

results = np.asmatrix(np.zeros((len(X),len(X.T))))

# estimate coefficients from day 64 onwards
for d in range(window,len(X)):
	X_sub = X[d-window:d+1,:] # only observations within most recent time window
	Y_sub = Y[d-window:d+1]
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

###############################################################################

# generate code to shade plot where phi is significant
def gen_axvspan(inds):
    for i in range(len(inds) - 1):
        start = inds[i]
        end = inds[i+1]
        if end - start == 1:
            print('ax.axvspan(x.loc[' + str(start) + '],x.loc[' + str(end) + '], facecolor=\'grey\', alpha = 0.25)')
        
###############################################################################

# exploratory analysis: durbl

asset = 'Durbl'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
durbl_CIs = pd.read_csv('output/Durbl/bwunif_bs_coeff_CIs_' + model + '.csv',index_col=0)

# date range where phi is significantly < 0
inds = list(durbl_CIs[durbl_CIs.ub < 0].index)

gen_axvspan(inds)

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
plt.savefig('figures/Durbl/bwunif_full_cumrets_ofint_' + model + '.jpg')


###############################################################################

# exploratory analysis: telcm

asset = 'Telcm'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
telcm_CIs = pd.read_csv('output/Telcm/bwunif_bs_coeff_CIs_' + model + '.csv',index_col=0)

# date range where phi is significantly > 0
inds = list(telcm_CIs[telcm_CIs.lb > 0].index)

gen_axvspan(inds)

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
plt.savefig('figures/Telcm/bwunif_full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# exploratory analysis: buseq

asset = 'BusEq'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
buseq_CIs = pd.read_csv('output/BusEq/bwunif_bs_coeff_CIs_' + model + '.csv',index_col=0)

# date range where phi is significantly < 0
inds = list(buseq_CIs[buseq_CIs.ub < 0].index)

gen_axvspan(inds)

# full-period cumulative returns with shaded area where phi > 0
full_cumulative_rets = np.cumprod(np.add(np.divide(Y.copy(),100),1)).tolist()[0]
x = data.loc[1:,'Date']

fig, ax = plt.subplots(figsize=(10,5))

line1, = ax.plot(x, full_cumulative_rets, color='r',label='cumulative return')
# axvspan code goes here
myFmt = mpld.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('Cumulative returns, BusEq, ' + model)

#plt.show()
plt.savefig('figures/BusEq/bwunif_full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# exploratory analysis: manuf

asset = 'Manuf'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
manuf_CIs = pd.read_csv('output/Manuf/bwunif_bs_coeff_CIs_' + model + '.csv',index_col=0)

# date range where phi is significantly > 0
inds = list(manuf_CIs[manuf_CIs.lb > 0].index)

gen_axvspan(inds)

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
plt.savefig('figures/Manuf/bwunif_full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# exploratory analysis: other

asset = 'Other'
model = 'FF5'
Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
other_CIs = pd.read_csv('output/Other/bwunif_bs_coeff_CIs_' + model + '.csv',index_col=0)

# date range where phi is significantly > 0
inds = list(other_CIs[other_CIs.lb > 0].index)

gen_axvspan(inds)

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
plt.savefig('figures/Other/bwunif_full_cumrets_ofint_' + model + '.jpg')

###############################################################################

# test a trading strategy

def forecast_ret(i,data,coeffs,cis,asset,model,strategy):
        forecast = np.add(coeffs.loc[i,'const'],np.multiply(coeffs.loc[i,'rm_rf'],data.loc[63+i,'Mkt-RF']))
        if model == 'FF3':
            forecast = np.add(forecast,np.multiply(coeffs.loc[i,'smb'],data.loc[63+i,'SMB']))
            forecast = np.add(forecast,np.multiply(coeffs.loc[i,'hml'],data.loc[63+i,'HML']))
        elif model == 'FF5':
            forecast = np.add(forecast,np.multiply(coeffs.loc[i,'smb'],data.loc[63+i,'SMB']))
            forecast = np.add(forecast,np.multiply(coeffs.loc[i,'hml'],data.loc[63+i,'HML']))
            forecast = np.add(forecast,np.multiply(coeffs.loc[i,'rmw'],data.loc[63+i,'RMW']))
            forecast = np.add(forecast,np.multiply(coeffs.loc[i,'cma'],data.loc[63+i,'CMA']))
        if strategy == 'phi':
            forecast += np.multiply((cis.loc[i,'lb'] > 0),np.multiply(coeffs.loc[i,'r_1'],np.subtract(data.loc[63+i,asset],data.loc[63+i,'RF'])))
            forecast += np.multiply((cis.loc[i,'ub'] < 0),np.multiply(coeffs.loc[i,'r_1'],np.subtract(data.loc[63+i,asset],data.loc[63+i,'RF'])))
        return forecast

def simulate_strategy(Y,data,coeffs,cis,asset,model,strategy):
    rets = [0] + (Y[63:].copy()).flatten().tolist()[0]
    rf = np.divide(data.loc[62:2516,'RF'].values,100)
    price = np.cumprod(np.add(np.divide(rets,100),1))
    borrowed = [1] + [0]*(len(rets)-1)
    sold = [0]*len(rets)
    position = ['long'] + ['']*(len(rets)-1)
    for i in range(len(coeffs)):
        borrowed[i+1] = borrowed[i]*np.exp(rf[i+1])
        sold[i+1] = sold[i]*np.exp(rf[i+1])
        r_tp1 = forecast_ret(i,data,coeffs,cis,asset,model,strategy)
        if r_tp1 > 0: # go long
            position[i+1] = 'long'
            if position[i] == 'short': # if not already long
                diff_to_borrow = price[i+1] - sold[i+1]
                if diff_to_borrow > 0: # if we need to borrow in order to make the purchase
                    sold[i+1] = 0 # use all of our money gained by short selling
                    borrowed[i+1] += diff_to_borrow # borrow the remainder rf
                if diff_to_borrow <= 0:
                    sold[i+1] -= price[i+1] # use some of our money gained from short selling
        elif r_tp1 < 0: # go short
            position[i+1] = 'short'
            if position[i] == 'long': # if not already short
                diff_to_invest = price[i+1] - borrowed[i+1]
                if diff_to_invest > 0: # if we have extra money leftover from the sale
                    borrowed[i+1] = 0
                    sold[i+1] += diff_to_invest
                if diff_to_invest <= 0:
                    borrowed[i+1] -= price[i+1] # pay off some of the borrowed amount
    results = pd.DataFrame({'ret_pct':rets,'rf_rate':rf,'position':position,'price':price,'borrowed':borrowed,'sold':sold})
    buyhold = price[-1] - np.cumprod(np.exp(rf))[-1]
    return [results,buyhold]

# make sure the correct data is loaded
data = clean_data('19940101','20031230')
strategy_results = pd.DataFrame({'buyhold_profit':[0]*3,'phi_profit':[0]*3,'pure_profit':[0]*3,},index = ['CAPM','FF3','FF5'])

for model in ['CAPM','FF3','FF5']:
    phi_profits = 0
    pure_profits = 0
    buyhold_profits = 0
    
    for asset in ['Durbl', 'Manuf','BusEq', 'Telcm','Other']:
        cis = pd.read_csv('output/' + asset + '/bwunif_bs_coeff_CIs_' + model + '.csv',index_col=0)
        coeffs = pd.read_csv('output/' + asset + '/bwunif_nonparam_coeff_ests_' + model + '.csv',index_col=0)
        Y = (np.matrix(np.subtract(data.loc[1:,asset].values,data.loc[1:,'RF'].values)).T)
    
        phi = simulate_strategy(Y,data,coeffs,cis,asset,model,'phi')
        phi_result = phi[0]
        pure = simulate_strategy(Y,data,coeffs,cis,asset,model,'pure')
        pure_result = pure[0]
        
        phi_profit = (phi_result.loc[2454,'position'] == 'long')*phi_result.loc[2454,'price'] + phi_result.loc[2454,'sold'] - phi_result.loc[2454,'borrowed']
        phi_profits += phi_profit
        
        pure_profit = (pure_result.loc[2454,'position'] == 'long')*pure_result.loc[2454,'price'] + pure_result.loc[2454,'sold'] - pure_result.loc[2454,'borrowed']
        pure_profits += pure_profit
        
        buyhold_profits += pure[1]
        
    strategy_results.loc[model,'buyhold_profit'] = buyhold_profits
    strategy_results.loc[model,'phi_profit'] = phi_profits
    strategy_results.loc[model,'pure_profit'] = pure_profits


















