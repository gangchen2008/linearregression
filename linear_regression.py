# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:18:46 2019

@author: root
"""

import pandas as pd
import numpy as np
import os
import talib as ta
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def generate_feature(df):
    base = 16
    res = np.ones((df.shape[0],10),dtype=np.float32)*np.nan
    fn = []
    
    H = df['high'].values * df['adjust_factor'].values
    L = df['low'].values * df['adjust_factor'].values
    M = df['mid'].values * df['adjust_factor'].values
    V = df['vol'].values.copy().astype(np.float64)
    oi = df['sectional_cjbs'].values.copy().astype(np.float64)
    
    cnt = 0
    
    for i in range(0,7):
        fast,slow = base * (2 ** i), base * 2 *(2 ** i)
        timeperiod = fast
        res[:, cnt] = ta.MA(M, fast)/ta.MA(M, slow)
        cnt += 1
        fn.append("({}MA-{}MA)".format(fast, slow))
        
    res = res[:, :cnt]
    return pd.DataFrame(res, index = df.index),fn

def generate_label(df):
    base = 16
    res = np.ones((df.shape[0], 1),dtype=np.float32)*np.nan
    fn = []
    
    M = df['mid'].values * df['adjust_factor'].values
    C = df['close'].values * df['adjust_factor'].values
    
    for i in range(0,(df.shape[0] - base)):
        res[i, 0] = math.log(M[i + base - 1]/M[i])
        
    fn.append("({}RETURN)".format(base))
        
    res = res[:, :1]
    return pd.DataFrame(res, index = df.index),fn  
    
if _name_ == '_main_':
    
    #step1:load data
    df = pd.read_hdf('C:/Users/root/Desktop/ZL000001.hdf',key = "table")
    
    # step 2: generate feature
    feature,feature_name = generate_feature(df)
    
    # step 3: generate label (future 16 minutes return)
    ret,ret_name = generate_label(df)
    
    # step 4: linear regression and get in-sample prediction result
    #将feature和ret中NA的行数进行删除，得到data_x,data_y
    data_all = pd.concat([feature,ret],axis=1,ignore_index=True)
    data_all_del_na =data_all.dropna()
    data_x = data_all_del_na.ix[:,0:6]
    data_y = data_all_del_na.ix[:,7]
    #利用tarin_test_split将训练集和测试集进行分开，test_size占30%
    X_train,X_test,y_train,t_test = train_test_split(data_x,data_y,test_size=0.3)
    #引入训练方法
    model = LinearRegression()
    model.fit(X_train,y_train)
    prediction = model.predict(X_train)
    model.score(X_train,y_train)
    
    # step 5: calculate 90% percentile for the abs prediction
    # try to use numpy.percentile
    percentile_90 = np.percentile(abs(prediction),90)
    
    # step 6: transform prediction into position
    prediction_test = model.predict(X_test)
    position = np.zeros((X_test.shape[0], 1),dtype=np.float32)
    position[prediction_test > percentile_90] = 1
    position[prediction_test < (percentile_90*(-1))] = -1