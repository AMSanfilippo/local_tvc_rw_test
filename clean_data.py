#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:47:12 2018

@author: as
"""

# file '12_Industry_Portfolios_Daily.csv': 
# daily returns on 12 industry portfolios from July 1, 1926 - December 31, 2017.
# assigns each NYSE, AMEX, and NASDAQ stock to an industry portfolio based on 
# its four-digit SIC code at that time.

# file 'F-F_Momentum_Factor_daily.csv':
# daily returns on a momentum portfolio from November 3, 1926 - December 31, 2017
# long high-momentum stocks, short low-momentum stocks

# file 'F-F_Research_Data_Factors_daily.csv':
# daily returns on three Fama-French factors from July 1, 1926 - December 31, 2017
# Rm-Rf = excess return on market 
# SMB = avg. return on portfolio long smallest stocks, short largest stocks
# HML = avg. return on portfolio long hi-value stocks, short lo-value stocks

import pandas as pd
import numpy as np
from datetime import datetime

# clean datasets for analysis
# start_date: first date in sample period, specify as string 'YYYYMMDD'
# start_date: last date in sample period, specify as string 'YYYYMMDD'
def clean_data(start_date,end_date):

    industry_ret = pd.read_csv('12_Industry_Portfolios_Daily.csv',nrows=24140)
    industry_ret = industry_ret.rename(columns={'Unnamed: 0':'Date'})
    
    momentum_ret = pd.read_csv('F-F_Momentum_Factor_daily.csv')
    momentum_ret = momentum_ret.rename(columns={'Unnamed: 0':'Date'})
    
    factors = pd.read_csv('F-F_Research_Data_Factors_daily.csv')
    factors = factors.rename(columns={'Unnamed: 0':'Date'})
    
    data = factors.merge(industry_ret,how='inner',on='Date')
    data = data.merge(momentum_ret,how='inner',on='Date')
    
    data['Date'] = [datetime.strptime(str(d),'%Y%m%d') for d in data['Date']]
    
    # truncate data to desired time period: Jan. 1 1988 - Dec. 30 2016
    data = data.loc[(data.Date >= datetime.strptime(start_date,'%Y%m%d')) & (data.Date <= datetime.strptime(end_date,'%Y%m%d')),:] 
    data = data.reset_index().drop(['index'],axis=1)

    # TO-DO: CHANGE RETURNS TO BASIS POINTS

    return data

# generate X matrix for regression, assuming AR(1) for returns
# asset: string
# factor_model: string, either 'CAPM' or 'FF3'
def gen_X(data,asset,factor_model):
    
    # constant and portfolio factors 
    factors = np.vstack([[1]*(len(data)-1),data.loc[1:,'Mkt-RF'].values])
    if factor_model == 'FF3':
        factors = np.vstack([factors,data.loc[1:,'SMB'].values,data.loc[1:,'HML'].values])

    # lag-one excess return 
    X = np.vstack([factors,np.subtract(data.loc[:len(data)-2,asset].values,data.loc[:len(data)-2,'RF'].values)])
    
    # make matrix and return
    X = np.matrix(X).T
    
    return X

    






