#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:28:54 2018

@author: as
"""

# obtain fitted values, residuals, and centered residuals from estimated model
# hypothesis: string, "null" or "alternative" (as of now this only handles alternative)
# X: independent variable matrix with column of 1s for intercept
# Y: dependent variable matrix
# coeffs: dataframe of fitted tvcs from estimation of alternative (tvc) model
# factor_model: string, either 'CAPM' or 'FF3'
# start: row index of first time value for which we have coeff. estimates (integer)
# end: row index of last time value for which we have coeff. estimates (integer)
def get_residuals(hypothesis,X,Y,coeffs,factor_model,start,end):
    # constant component of fitted values
    fit_const = coeffs['const'].values
    # market return component of fitted values
    fit_rmrf = [a*b for a,b in zip(X[start:end,1].flatten().tolist()[0],coeffs['rm_rf'].values)] 
    # lag-one return component of fitted values
    fit_r_1 = [a*b for a,b in zip(X[start:end,-1].flatten().tolist()[0],coeffs['r_1'].values)]

    fit_smb = [0]*len(X[start:end])
    fit_hml = [0]*len(X[start:end])

    if factor_model == 'FF3':
        fit_smb = [a*b for a,b in zip(X[start:end,2].flatten().tolist()[0],coeffs['smb'].values)]
        fit_hml = [a*b for a,b in zip(X[start:end,3].flatten().tolist()[0],coeffs['hml'].values)]
    

    fitted = [a+b+c+d+e for a,b,c,d,e in zip(fit_const,fit_rmrf,fit_r_1,fit_smb,fit_hml)]
    
    resids = [a-b for a,b, in zip(Y[start:end].flatten().tolist()[0],fitted)]
    
    mean_resid = np.mean(resids)
    centered_resids = [r-mean_resid for r in resids]
    
    return pd.DataFrame({'fitted':fitted,'resids':resids,'centered_resids':centered_resids})

 
# resample with replacement from {resids}
def bs_resample(resids):
    bs_resids = np.random.choice(resids,len(resids),replace=True)
    return bs_resids
 
    
# recursively generate new sample {(y*,x)} using bs sample of residuals
# hypothesis: string, "null" or "alternative" (as of now this only handles alternative)
# coeffs: dataframe of fitted tvcs from estimation of alternative (tvc) model
# X: full X matrix used for initial estimation 
# resids: bootstrap sample of residuals from alternative model estimation
# factor_model: string, either 'CAPM' or 'FF3'
# start: row index of first time value for which we have coeff. estimates (integer)
# end: row index of last time value for which we have coeff. estimates (integer)
def gen_recursive(hypothesis,coeffs,X,resids,factor_model,start,end):
    r_0 = X[start,-1] # lagged return value to serve as "seed" for DGP
    
    # not really sure how to deal with the DGP outside of the timeframe for which we have coeff. estimates...
    # for now, let's just use the original outcome values...and maybe add some residual noise later? (Need to ask about this)
    X_sub = X[start:end]
    
    # initialize empty list of new returns
    r_star = [0]*len(X_sub)
    
    # generate r* values
    for row in range(len(X_sub)): 
        if row == 0:
            lagged_r = r_0
        else:
            lagged_r = r_star[row-1]
        current_r = coeffs.loc[row,'const'] + (coeffs.loc[row,'rm_rf']*X_sub[row,1]) + (coeffs.loc[row,'r_1']*lagged_r) + resids[row]
        if factor_model == 'FF3':
            current_r = current_r + (coeffs.loc[row,'smb']*X_sub[row,2]) + (coeffs.loc[row,'hml']*X_sub[row,3])
        r_star[row] = current_r
    
    # replace lagged return values in X with r*
    X_sub[1:,-1] = np.matrix(r_star[:len(r_star)-1]).T
    Y_star = np.matrix(r_star).T
    
    return [X,Y_star]
    
    
    
    