#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:28:54 2018

@author: as
"""

import numpy as np
import pandas as pd

# obtain fitted values, residuals, and centered residuals from estimated model
# hypothesis: string, "null" or "alternative" (as of now this only handles alternative)
# X: independent variable matrix with column of 1s for intercept
# Y: dependent variable matrix
# coeffs: dataframe of fitted tvcs from estimation of alternative (tvc) model
# factor_model: string, either 'CAPM' or 'FF3'
def get_residuals(hypothesis,X,Y,coeffs=pd.DataFrame(),factor_model='CAPM'):
    
    if hypothesis == 'alternative':
        # constant component of fitted values
        fit_const = coeffs['const'].values
        # market return component of fitted values
        fit_rmrf = np.multiply(np.array(X[:,1].T)[0],coeffs['rm_rf'].values)
        # lag-one return component of fitted values
        fit_r_1 = np.multiply(np.array(X[:,-1].T)[0],coeffs['r_1'].values) 
    
        fit_smb = np.array([0]*len(X))
        fit_hml = np.array([0]*len(X))
    
        if factor_model == 'FF3':
            fit_smb = np.multiply(np.array(X[:,2].T)[0],coeffs['smb'].values) 
            fit_hml = np.multiply(np.array(X[:,3].T)[0],coeffs['hml'].values)
    
        fitted = np.add(np.add(np.add(np.add(fit_const,fit_rmrf),fit_r_1),fit_smb),fit_hml)
    
    if hypothesis == 'null':
        fitted = np.array((X.dot(np.linalg.solve((X.T).dot(X),(X.T).dot(Y)))).T)[0]

    resids = np.subtract(np.array(Y.T)[0],fitted)
        
    mean_resid = np.mean(resids)
    centered_resids = np.subtract(resids,mean_resid)
    
    return pd.DataFrame({'fitted':fitted,'resids':resids,'centered_resids':centered_resids})

 
# resample with replacement from {resids}
def bs_resample(resids):
    bs_resids = np.random.choice(resids,len(resids),replace=True)
    return bs_resids
 
    
# recursively generate new sample {(y*,x)} using bs sample of residuals
# hypothesis: string, "null" or "alternative" (as of now this only handles alternative)
# coeffs: dataframe of fitted tvcs from estimation of alternative (tvc) model
# X: c(opy of) X matrix used for initial estimation
# r_0: lagged return value to serve as "seed" for DGP
# resids: bootstrap sample of residuals from alternative model estimation
# factor_model: string, either 'CAPM' or 'FF3'
def gen_recursive(hypothesis,X,r_0,resids,coeffs,factor_model='CAPM'):
        
    # initialize empty list of new returns
    r_star = [0]*len(X)
    
    # generate r* values under alternative hypothesis
    for row in range(len(X)): 
        if row == 0:
            lagged_r = r_0
        else:
            lagged_r = r_star[row-1]
        
        if hypothesis == 'alternative':
            current_r = np.add(np.add(np.add(coeffs.loc[row,'const'],np.multiply(coeffs.loc[row,'rm_rf'],X[row,1])),np.multiply(coeffs.loc[row,'r_1'],lagged_r)),resids[row])
            if factor_model == 'FF3':
                current_r = np.add(np.add(current_r,np.multiply(coeffs.loc[row,'smb'],X[row,2])),np.multiply(coeffs.loc[row,'hml'],X[row,3]))
    
        if hypothesis == 'null':
            current_r = np.add(np.add(np.add(coeffs[0],np.multiply(coeffs[1],X[row,1])),np.multiply(coeffs[-1],lagged_r)),resids[row])
            if factor_model == 'FF3':
                current_r = np.add(np.add(current_r,np.multiply(coeffs[2],X[row,2])),np.multiply(coeffs[3],X[row,3]))
        
        r_star[row] = current_r
    
    # return recursively-generated returns and new lagged return column
    Y_star = np.matrix(r_star).T
    X_star = np.matrix([r_0] + r_star[:len(r_star)-1]).T
    
    return [X_star,Y_star]
      
