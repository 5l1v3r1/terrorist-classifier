#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:30:18 2018

@author: richard
"""

import sys
import numbers
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

def import_data(variable_list=None):
    print('Reading data...')
    with warnings.catch_warnings(): #ignore DtypeWarning
        warnings.simplefilter('ignore')
        df_full = pd.read_csv('/Users/richard/Documents/Fellowship.ai/GTD_0617dist/globalterrorismdb_0617dist.csv')
    if variable_list:
        cols = pd.read_csv(variable_list)
    #select provided features
    cols = list(cols['headers'])
    df = df_full[cols]
    #filter out samples with unknown attackers
    df = df[df['gname'] != 'Unknown']
    return df

# For a categorical variable, aggregates low-frequency values to reduce the
# number of dummy variables in subsequent one-hot encoding.
def aggregate(df_or, col, threshold=1):
    df = df_or[df_or[col].notnull()]
    print('One-hot: ' + col)
    set_vals = list(set(df[col]))
    if (len(set_vals) < 30):     # slice df
        vals = {v: sum(df[col] == v) for v in set_vals}
    else:                   # element-wise
        vals = {v:0 for v in set_vals}
        for index, row in df.iterrows():
            vals[row[col]] += 1
    
    used, unused, unknown = {}, {}, {}
    if len(df_or) - len(df) > 0:
        unknown['NaN'] = len(df_or) - len(df)
    for k,v in vals.items():
        if isinstance(k, numbers.Real) and k < 0:
            unknown[k] = v
        else:
            if v < threshold:
                unused[k] = v
            else:
                used[k] = v
    #print(sum(unused.values())/len(df_or))
    return used, unused, unknown

#convert categorical variables to binary variables
def one_hot(df, col, used, unused, unknown):
    for val in used:
        df[col + '_' + str(val)] = (df[col] == val)
    if unused:
        df[col + '_other'] = df[col].isin(unused)
    if unknown:
        df[col + '_unknown'] = (df[col].isin(unknown) | df[col].isnull())
    
    # drop last column since only need k-1 dummy variables
    df.drop([col, df.columns[-1]], axis=1, inplace=True)
    return df

# For a scalar variable, flags all unknown observations in a new column and 
# sets unknown observations to zero.
def clean_scalar(df, col, special_unknowns):
    unknown_rows = (df[col].isnull()) | (df[col] < 0) | \
                 (col in special_unknowns and df[col] == special_unknowns[col])
    if any(unknown_rows):        
        df[col + '_unknown'] = unknown_rows
        df.loc[unknown_rows, col] = 0
    return df
            
            
def run_model(train_x, train_y, test_x, test_y, detailed_metrics=False):    
    print('Scaling data...')
    scalers = StandardScaler()
    scalers.fit(train_x)
    trainx = scalers.transform(train_x)
    testx = scalers.transform(test_x)
    print('Running classifier...')
    model = MLPClassifier(hidden_layer_sizes=(20,20))
    model.fit(trainx, train_y)
    pred = model.predict(testx)
    with warnings.catch_warnings(): #ignore UndefinedMetricWarning
        warnings.simplefilter('ignore')
        metrics = precision_recall_fscore_support(test_y, pred, average='weighted') 
        if detailed_metrics:
            print(classification_report(test_y, pred))
    return metrics

import time
"""
This function

"""
def classify(group_thres=5, cate_thres=100, train_index=None, test_index=None, 
             df=None):
    start = time.time()
    print('\n\n {} {}'.format(group_thres, cate_thres))
    
    if df is None:
        df = import_data(variable_list = 
                         '/Users/richard/Documents/Fellowship.ai/variables LONGER.txt')
    
    #split train and test sets
    dfx = df.drop('gname', axis=1)
    dfy = df['gname']
    if train_index is not None:  # k-fold validation
        train_x = dfx.loc[train_index]
        train_y = dfy.loc[train_index]
        test_x = dfx.loc[test_index]
        test_y = dfy.loc[test_index]
    else:          # random train/test split
        train_x, test_x, train_y, test_y = train_test_split(dfx, dfy)
    
    #identify attackers and # of attacks
    groups = sorted(list(set(train_y)))
    groups = {g: 0 for g in groups}
    for entry in train_y:
        groups[entry] += 1
            
    #filter data by groups with 100+ attacks
    newgroups = []
    for group, count in groups.items():
        if count >= group_thres:
            newgroups.append(group)    
    train_x = train_x[train_y.isin(newgroups)]
    train_y = train_y[train_y.isin(newgroups)]
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    print('All data loaded')
    scalars = ['nperps', 'nkill', 'nwound', 'nhostkid', 'nhours', 'ndays', 'ransomamt']
    special_unknowns = {'nhours': 999}
    for col in scalars:
        train_x = clean_scalar(train_x, col, special_unknowns)
        test_x = clean_scalar(test_x, col, special_unknowns)
    categories = [c for c in list(df) if c not in scalars]
    for col in categories[1:]:
        # create these lists based on training set, then apply to both sets
        used, unused, unknown = aggregate(train_x, col, cate_thres)
        one_hot(train_x, col, used, unused, unknown)
        one_hot(test_x, col, used, unused, unknown)
    
    if train_index is None: detailed_metrics=True
    else: detailed_metrics=False
    metrics = run_model(train_x, train_y, test_x, test_y, detailed_metrics)
    print(time.time() - start)
    return metrics

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        
def cross_validate(group_thres=5, cate_thres=100, k=4):
    with SuppressPrints():
        # only read in data once
        df = import_data(variable_list = 
                         '/Users/richard/Documents/Fellowship.ai/variables LONGER.txt')
        df.reset_index(drop=True, inplace=True) # for kf split
        kfolds = KFold(n_splits=k, shuffle=True)
        splits = kfolds.split(df)
        metrics = []
        for train_index, test_index in splits:
            metric = classify(group_thres, cate_thres, train_index, test_index, df)
            metrics.append(metric)
        
        scores = pd.DataFrame(columns=['precision','recall','fscore'])
        scores.loc['average'] = [np.mean(m) for m in list(zip(*metrics))[:-1]]
        scores.loc['std'] = [np.std(m) for m in list(zip(*metrics))[:-1]]
    print(scores)
    return(scores)

#classify(100,100)
scores = cross_validate(500,500)