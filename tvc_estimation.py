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

# symmetric uniform weights until Gauss kernel is fixed
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


# function to perform WLS regression
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
X_test = gen_X(data,asset,'CAPM')
Y_test = (np.matrix(data.loc[1:,asset].values).T)

n = len(Y_test)
h_test = 0.1
start = math.floor((h*n)/2)
end = n - start

datelist_test = list(range(len(Y_test)))[start:end]

# run for loop in parallel
# with 8 cores and about 10 years of data, this runs in ~ 25 secs
# --> 500 bs replications should take ~ 35 mins
results = Parallel(n_jobs=numcores)(delayed(wls)(X=X_test,Y=Y_test,target=date,h=h_test,wt_fxn='uniform') for date in datelist_test)

coeff_results_df = pd.DataFrame(results)
coeff_results_df.columns = ['const','rm_rf','r_1']


# plot fitted values
fit_const = [a*b for a,b in zip(X_test[start:end,0].flatten().tolist()[0],coeff_results_df['const'].values)]
fit_rmrf = [a*b for a,b in zip(X_test[start:end,1].flatten().tolist()[0],coeff_results_df['rm_rf'].values)] 
fit_r_1 = [a*b for a,b in zip(X_test[start:end,2].flatten().tolist()[0],coeff_results_df['r_1'].values)]

fitted = [a+b+c for a,b,c in zip(fit_const,fit_rmrf,fit_r_1)]

x = range(1,len(datelist_test)+1)

fig, ax = plt.subplots()

line1, = ax.plot(x[:100], fitted[:100], dashes = [5,5], linewidth=2,color='b',label='fitted')
line2, = ax.plot(x[:100], Y_test[start:start+100], linewidth=2,color='r')

plt.show()


