#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:03:00 2018

@author: soudis
"""
import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold, KFold
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, f1_score, r2_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import randint as sp_randint, uniform as sp_unif, sem as npsem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from skopt import BayesSearchCV
from sklearn.model_selection import learning_curve
from lightgbm import LGBMRegressor, LGBMClassifier

# Return lists of models
def get_models(proba = False, classif = True, random_state = 42):
    """
    Returns a list of algorithms with basic parameters to be passed on for CV
    by CValidator and a list of their names (initialised)
    
    params:
    - proba: if the scoring function will need probability predictions or not
    - random_state: seed
    
    """
    if classif == False:
        return [RandomForestRegressor(random_state = random_state, n_jobs =-1, n_estimators = 500), Pipeline([('standartize', RobustScaler()), ('LSVR', LinearSVR(random_state= random_state, loss = 'epsilon_insensitive'))]),
               Pipeline([('standartize', RobustScaler()), ('SVR', SVR())]), Pipeline([('standartize', RobustScaler()), ('ElNet', ElasticNet(random_state = random_state))]),
               Pipeline([('standartize', RobustScaler()), ('KNN', KNeighborsRegressor(n_jobs = -1))]), LGBMRegressor(n_jobs=-1, random_state= random_state)], ['RF', 'LSVR', 'SVR', 'ElNet', 'KNN', 'LGB']
    
    if proba == False:
        return [RandomForestClassifier(random_state = random_state, n_jobs =-1, n_estimators = 500), Pipeline([('standartize', RobustScaler()), ('LSVC', LinearSVC(random_state= random_state, dual = False, loss = 'squared_hinge'))]),
               Pipeline([('standartize', RobustScaler()), ('SVC', SVC(random_state= random_state))]), Pipeline([('standartize', RobustScaler()), ('Logit', LogisticRegression(random_state = random_state, n_jobs = -1, solver = 'saga', max_iter = 5000))]),
               Pipeline([('standartize', RobustScaler()), ('KNN', KNeighborsClassifier(n_jobs = -1))]), LGBMClassifier(n_jobs=-1, random_state=random_state)], ['RF', 'LSVC', 'SVC', 'Logit', 'KNN', 'LGB']
    else:
        return [RandomForestClassifier(random_state = random_state, n_jobs =-1, n_estimators = 500), Pipeline([('standartize', RobustScaler()), ('SVC', SVC(kernel = 'linear', random_state= random_state, probability=True))]),
               Pipeline([('standartize', RobustScaler()), ('SVC', SVC(random_state= random_state, probability=True))]), Pipeline([('standartize', RobustScaler()), ('Logit', LogisticRegression(random_state = random_state, n_jobs = -1, solver = 'saga', max_iter = 5000))]),
               Pipeline([('standartize', RobustScaler()), ('KNN', KNeighborsClassifier(n_jobs = -1))]), LGBMClassifier(n_jobs=-1, random_state= random_state)], ['RF', 'LSVC', 'SVC', 'Logit', 'KNN', 'LGB']


# Validator with ttest
def CValidator(models, X_trains, y_train, clf_names, cv = RepeatedStratifiedKFold(10,3, random_state = 42),
               scoring = 'accuracy'):
    """
    Given a list of algorithms and different subsambles of X_train, this will crossvalidate
    the performance of the models and return an order pd.DataFrame containing the CV score,
    its Standard Deviation, and a t-tests testing the best models score against all other models. 
    
    params:
    - models : a list of algorithms to train
    - X_trains : a list of data to train the algorithms on
    - y_train : train labels
    - clf_names: list of classifiers names
    - cv : type of CV to be used
    - random_state : random state seed
    - scoring: the metric to be used for CV scoring
    """
    results = {}
    tests_data = {}
    
    if scoring == 'neg_mean_squared_error':
        
        for i in range(0, len(models)):
            for j in range(0, len(X_trains)):
                score = cross_val_score(models[i], X_trains[j],
                                        y_train,
                                        n_jobs=-1,
                                        cv = cv,
                                        scoring=scoring)
                results.setdefault((clf_names[i] + '_' + str(j)),  {'Score':np.sqrt(-score.mean()),'S.E.': npsem(np.sqrt(-score))})
                tests_data.setdefault((clf_names[i] + '_' + str(j)), -score)
                print('Finished CV for' + ' ' + (clf_names[i] + ' ' + str(j)))
        results = pd.DataFrame(results).T.sort_values(by = ['Score'], ascending=True)
        
        p_values = np.zeros(len(results))
        for i in range(0, len(tests_data)):
            p_values[i] = sps.ttest_rel(tests_data[results.iloc[0,:].name], tests_data[results.iloc[i,:].name])[1]
                
        results['p_val'] = pd.Series(p_values, index=results.index)
        return results.sort_values(['Score', 'p_val'], ascending=[True, False]).applymap(lambda x: '%.4f' % x)
    
    
    if scoring in ['neg_log_loss', 'neg_mean_absolute_error',
                   'neg_median_absolute_error', 'neg_mean_absolute_error']:
        for i in range(0, len(models)):
            for j in range(0, len(X_trains)):
                score = cross_val_score(models[i], X_trains[j],
                                        y_train,
                                        n_jobs=-1,
                                        cv = cv,
                                        scoring=scoring)
                results.setdefault((clf_names[i] + '_' + str(j)),  {'Score':-score.mean(),'S.E.':npsem(-score)})
                tests_data.setdefault((clf_names[i] + '_' + str(j)), -score)
                print('Finished CV for' + ' ' + (clf_names[i] + ' ' + str(j)))
        results = pd.DataFrame(results).T.sort_values(by = ['Score'], ascending=True)
        
        p_values = np.zeros(len(results))
        for i in range(0, len(tests_data)):
            p_values[i] = sps.ttest_rel(tests_data[results.iloc[0,:].name], tests_data[results.iloc[i,:].name])[1]
                
        results['p_val'] = pd.Series(p_values, index=results.index)
        return results.sort_values(['Score', 'p_val'], ascending=[True, False]).applymap(lambda x: '%.4f' % x)
    else:
        for i in range(0, len(models)):
            for j in range(0, len(X_trains)):
                score = cross_val_score(models[i], X_trains[j],
                                        y_train,
                                        n_jobs=-1,
                                        cv = cv,
                                        scoring=scoring)
                results.setdefault((clf_names[i] + '_' + str(j)),  {'Score': score.mean(),'S.E.':npsem(score)})
                tests_data.setdefault((clf_names[i] + '_' + str(j)), score)
                print('Finished CV for' + ' ' + (clf_names[i] + ' ' + str(j)))
        results = pd.DataFrame(results).T.sort_values(by = ['Score'], ascending=False)
      
        p_values = np.zeros(len(results))
        for i in range(0, len(tests_data)):
            p_values[i] = sps.ttest_rel(tests_data[results.iloc[0,:].name], tests_data[results.iloc[i,:].name])[1]
                
        results['p_val'] = pd.Series(p_values, index=results.index)
        return results.sort_values(['Score', 'p_val'], ascending=[False, False]).applymap(lambda x: '%.4f' % x)


# return dict of parameters for RandomGridCV
def get_params_rand(proba = False, classif = True):
    """
    Returns a list of parameters with basic parameters to be passed on for tuning
    by the RandomCV
    
    """
    if classif == False:
        return [{'max_depth': sp_randint(1, 50), 'min_samples_leaf': sp_randint(2, 20), 'max_features': sp_expon.rvs(0.03, 0.11)},
               {'LSVR__C': sp_expon(scale = 100)},
               {'SVR__C': sp_expon(scale = 100), 'SVR__gamma': sp_expon(scale = 0.1)},
               {'ElNet__alpha': sp_expon(scale = 100), 'ElNet__l1_ratio': sp_unif(0, 0.999)},
               {'KNN__n_neighbors': sp_randint(2, 30), 'KNN__weights': ['uniform', 'distance']},
               {'LGBRagressor': }
               ]
        
    if proba == False:
        return [{'max_depth': sp_randint(1, 50), 'min_samples_leaf': sp_randint(2, 20), 'max_features': sp_expon.rvs(0.03, 0.11)},
               {'LSVC__C': sp_expon(scale = 100), 'LSVC__penalty': ['l2', 'l1']},
               {'SVC__C': sp_expon(scale = 100), 'SVC__gamma': sp_expon(scale = 0.1)},
               {'Logit__penalty': ['l2', 'l1'], 'Logit__C': sp_expon(scale = 100)},
               {'KNN__n_neighbors': sp_randint(2, 30), 'KNN__weights': ['uniform', 'distance']},
               ]
    else:
        return [{'max_depth': sp_randint(1, 50), 'min_samples_leaf': sp_randint(2, 20), 'max_features': sp_expon.rvs(0.03, 0.11)},
               {'LSVC__C': sp_expon(scale = 100)},
               {'SVC__C': sp_expon(scale = 100), 'SVC__gamma': sp_expon(scale = 0.1)},
               {'Logit__penalty': ['l2', 'l1'], 'Logit__C': sp_expon(scale = 100)},
               {'KNN__n_neighbors': sp_randint(2, 30), 'KNN__weights': ['uniform', 'distance']},
               ]
        
def get_params_skopt(proba = False, classif = True):
    """
    Returns a list of parameters with basic parameters to be passed on for tuning
    by the BayesCV
    
    """
    if classif == False:
        return [{'max_depth': (1, 50, 'uniform'), 'min_samples_leaf': (2, 20, 'uniform'), 'max_features': (0.03, 0.11, 'log_uniform')},
               {'LSVR__C': (0.001, 1e+3, 'log-uniform')},
               {'SVR__C': (0.001, 1e+3, 'log-uniform'), 'SVR__gamma': (0.01, 0.3, 'log-uniform')},
               {'ElNet__alpha': (0.001, 1e+3, 'log-uniform'), 'ElNet__l1_ratio': (0, 1, 'log-uniform')},
               {'KNN__n_neighbors': (2, 30), 'KNN__weights': ['uniform', 'distance']},
               ]
        
    if proba == False:
        return [{'max_depth': (1, 50, 'uniform'), 'min_samples_leaf': (2, 20, 'uniform'), 'max_features': (0.03, 0.11, 'log_uniform')},
               {'LSVC__C': (0.001, 1e+3, 'log-uniform'), 'LSVC__penalty': ['l2', 'l1']},
               {'SVC__C': (0.001, 1e+3, 'log-uniform'), 'SVC__gamma': (0.01, 0.3, 'log-uniform')},
               {'Logit__penalty': ['l2', 'l1'], 'Logit__C': (0.001, 1e+3, 'log-uniform')},
               {'KNN__n_neighbors': (2, 30), 'KNN__weights': ['uniform', 'distance']},
               ]
    else:
        return [{'max_depth': (1, 50, 'uniform'), 'min_samples_leaf': (2, 20, 'uniform'), 'max_features': (0.03, 0.11, 'log_uniform')},
               {'LSVC__C': (0.001, 1e+3, 'log-uniform')},
               {'SVC__C': (0.001, 1e+3, 'log-uniform'), 'SVC__gamma': (0.001, 0.3, 'log-uniform')},
               {'Logit__penalty': ['l2', 'l1'], 'Logit__C': (0.001, 1e+3, 'log-uniform')},
               {'KNN__n_neighbors': (2, 30), 'KNN__weights': ['uniform', 'distance']},
               ]


## Perform (nested)  Random CV search
def NestedRandomizedcv(tups, y_train, cv_outer = StratifiedKFold(10, random_state=42),
                       cv_inner = StratifiedKFold(10, random_state=42), n_iter = 60,
                       nested = True, random_state = 42, scoring='accuracy'):
    """
    Given a tuple of algorithms/different datasets/names/parameters this will
    perform randomised cross validation using nested CV to choose the best parameters for
    each combination of model and datset.
    
    params:
    - tups : a list of tuples containing algorithm/datasets/names/parameters 
    - y_train : train labels
    - cv_outer : type of CV to be used for the outer validation
    - cv_inner : type of CV to be used for the inner validation, i.e., hyperparameter
                 optimization
    - nested : whether to performe nested CV
    - n_iter: number of iterations for the randomized search
    - random_state: seed for the randomized search
    - scoring: scoring metric to be used
    """
    results = []
    
    if nested == True:
        
        for i in range(0, len(tups)):
            for j in range(0, len(tups[i][1])):
                for train, test in cv_outer.split(tups[i][1][j], y_train):
                    tX_train = tups[i][1][j][train]
                    ty_train = y_train[train]
                    tX_test = tups[i][1][j][test]
                    ty_test = y_train[test]
                
                
                    Rcv = RandomizedSearchCV(tups[i][0], tups[i][3], cv = cv_inner,
                                             random_state = random_state, scoring=scoring,
                                             n_jobs=-1, n_iter = n_iter)
                    Rcv.fit(tX_train, ty_train)
                
                    results.append({'Combo':tups[i][2][j], 'Best_Params': Rcv.best_params_,
                                    'Outer_CV': Rcv.score(tX_test, ty_test),
                                    'Inner_CV': Rcv.best_score_})
            
                print('Finished with' + ' ' + tups[i][2][j])
    
        results = pd.DataFrame(results)
        results['Best_Params'] = results['Best_Params'].apply(str)
    
        if scoring == 'neg_mean_squared_error':
            
            results['Inner_CV'] = np.sqrt(-results['Inner_CV'])
            results['Outer_CV'] = np.sqrt(-results['Outer_CV'])
        
        if scoring in ['neg_log_loss', 'neg_mean_absolute_error',
                       'neg_median_absolute_error', 'neg_mean_absolute_error']:
            
            results['Inner_CV'] = -results['Inner_CV']
            results['Outer_CV'] = -results['Outer_CV']
    
        grouped = results.groupby(['Combo', 'Best_Params'])
        to_aggregate = {'Inner_CV':['mean', 'sem'], 'Outer_CV':['mean', 'sem', 'count']}
        results = grouped.agg(to_aggregate)
    
    else:
        for i in range(0, len(tups)):
            for j in range(0, len(tups[i][1])):
                
                Rcv = RandomizedSearchCV(tups[i][0], tups[i][3], cv = cv_inner,
                                         random_state = random_state, scoring=scoring,
                                         n_jobs=-1, n_iter = n_iter)
                Rcv.fit(X_train, y_train)
                
                results.append({'Combo':tups[i][2][j], 'Best_Params': Rcv.best_params_,
                                   'Score_CV': Rcv.best_score_})
            
                print('Finished with' + ' ' + tups[i][2][j])
    
        results = pd.DataFrame(results)
        results['Best_Params'] = results['Best_Params'].apply(str)
    
        if scoring == 'neg_mean_squared_error':
            
            results['Score_CV'] = np.sqrt(-results['Score_CV'])
        
        if scoring in ['neg_log_loss', 'neg_mean_absolute_error',
                       'neg_median_absolute_error', 'neg_mean_absolute_error']:
            
            results['Score_CV'] = -results['Score_CV']
    
        grouped = results.groupby(['Combo', 'Best_Params'])
        to_aggregate = {'Score_CV':['mean', 'sem', 'count']}
        results = grouped.agg(to_aggregate)
        
    return results

## Perform (nested)  Skopt CV search
def NestedBayesCV(tups, y_train, cv_outer = StratifiedKFold(10, random_state=42),
                       cv_inner = StratifiedKFold(10, random_state=42), nested = True,
                       n_iter = 60, random_state = 42, scoring='accuracy'):
    """
    Given a tuple of algorithms/different datasets/names/parameters this will
    perform randomised cross validation using nested CV to choose the best parameters for
    each combination of model and datset.
    
    params:
    - tups : a list of tuples containing algorithm/datasets/names/parameters 
    - y_train : train labels
    - cv_outer : type of CV to be used for the outer validation
    - cv_inner : type of CV to be used for the inner validation, i.e., hyperparameter
                 optimization
    - nested : whether to performe nested CV
    - n_iter: number of iterations for the randomized search
    - random_state: seed for the randomized search
    - scoring: scoring metric to be used
    """
    results = []
    
    if nested == True:
        
        for i in range(0, len(tups)):
            for j in range(0, len(tups[i][1])):
                for train, test in cv_outer.split(tups[i][1][j], y_train):
                    tX_train = tups[i][1][j][train]
                    ty_train = y_train[train]
                    tX_test = tups[i][1][j][test]
                    ty_test = y_train[test]
                    
                
                    opt = BayesSearchCV(tups[i][0], tups[i][3], cv = cv_inner,
                                         random_state = random_state, scoring=scoring,
                                         n_jobs=-1, n_iter = n_iter)
                
                    opt.fit(tX_train, ty_train)
                
                    results.append({'Combo':tups[i][2][j], 'Best_Params': opt.best_params_,
                                   'Outer_CV': opt.score(tX_test, ty_test),
                                   'Inner_CV': opt.best_score_})
            
                print('Finished with' + ' ' + tups[i][2][j])
    
        results = pd.DataFrame(results)
        results['Best_Params'] = results['Best_Params'].apply(str)
    
        if scoring == 'neg_mean_squared_error':
            
            results['Inner_CV'] = np.sqrt(-results['Inner_CV'])
            results['Outer_CV'] = np.sqrt(-results['Outer_CV'])
        
        if scoring in ['neg_log_loss', 'neg_mean_absolute_error',
                       'neg_median_absolute_error', 'neg_mean_absolute_error']:
            
            results['Inner_CV'] = -results['Inner_CV']
            results['Outer_CV'] = -results['Outer_CV']
    
        grouped = results.groupby(['Combo', 'Best_Params'])
        to_aggregate = {'Inner_CV':['mean', 'sem'], 'Outer_CV':['mean', 'sem', 'count']}
        results = grouped.agg(to_aggregate)
    
    else:
        for i in range(0, len(tups)):
            for j in range(0, len(tups[i][1])):
                opt = BayesSearchCV(tups[i][0], tups[i][3], cv = cv_inner,
                                         random_state = random_state, scoring=scoring,
                                         n_jobs=-1, n_iter = n_iter)
                
                opt.fit(X_train, y_train)
                results.append({'Combo':tups[i][2][j], 'Best_Params': opt.best_params_,
                                   'Score_CV': opt.best_score_})
        
                print('Finished with' + ' ' + tups[i][2][j])
                
        results = pd.DataFrame(results)
        results['Best_Params'] = results['Best_Params'].apply(str)
    
        if scoring == 'neg_mean_squared_error':
            
            results['Score_CV'] = np.sqrt(-results['Score_CV'])
        
        if scoring in ['neg_log_loss', 'neg_mean_absolute_error',
                       'neg_median_absolute_error', 'neg_mean_absolute_error']:
            
            results['Score_CV'] = -results['Score_CV']
    
        grouped = results.groupby(['Combo', 'Best_Params'])
        to_aggregate = {'Score_CV':['mean', 'sem', 'count']}
        results = grouped.agg(to_aggregate)
    
    return results


def get_test_score(models, trains, tests, y_train, y_test, names, random_state = 42, scoring = 'accuracy'):
    """
    Given a list of models, a list of train datasets, and a list of tests datasets, this will
    score the different algorithms agains the tests data and return a dataframe with the scores
    
    params:
    - models : a list of models to be trained and scored 
    - trains : a list with train data
    - y_train : train labels
    - tests : a list with test data
    - y_test: test labels
    - random_state: seed for the randomized search
    - scoring: scoring metric to be used, available are: accuracy, log_loss, f1, rmse, mae, r2 and auc
    """
    
    results = {}
    
    for i in range(0, len(models)):
        for j in range(0, len(trains)):
            models[i].fit(trains[j], y_train)
            
            if scoring == 'accuracy':
                score = accuracy_score(y_test, models[i].predict(tests[j]))
            
            elif scoring == 'log_loss':
                score = log_loss(y_test, models[i].predict_proba(tests[j]))
            
            elif scoring == 'f1':
                score = f1_score(y_test, models[i].predict(tests[j]))
                
            elif scoring == 'rmse':
                score = np.sqrt(mean_squared_error(y_test, models[i].predict(tests[j])))
            
            elif scoring == 'mae':
                score = mean_absolute_error(y_test, models[i].predict(tests[j]))
                
            elif scoring == 'r2':
                score = r2_score(y_test, models[i].predict(tests[j]))
                
            elif scoring == 'auc':
                score = roc_auc_score(y_test, models[i].predict(tests[j]))
            
            results.setdefault((names[i] + '_' + str(j)), {'Test Score': score})
            print('Finished Scoring' + ' ' + names[i] + '_' + str(j))
    
    return pd.DataFrame(results).T

# NearZeroVar
def near_zero_var(df, freqCut = 95/5, uniqueCut = 10):
    """
    Stolen from R's caret: Given a numeric pd.DataFrame or np.ndarray, this will return 
    a DataFrame whith variables having zero or near-zero variance
    omitted. The parameters freqCut and uniqueCut determine the 
    cut-off point for the near-zero variables.
    
    params:
    - df : pd.DataFrame or np.ndarray
    - freqCut : the cutoff for the ratio of the most common value
                to the second most common value
    - uniqueCut : the cutoff for the percentage of distinct values
                  out of the number of total samples
    """
    if not isinstance(df, pd.DataFrame) and not isinstance(df, np.ndarray):
        print('Input must be a Pandas or Numpy object')
    
    elif isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
        df = df.drop(df.columns[df.apply(lambda x: len(x.unique()), axis = 0) <2], axis = 1)
        LowCut = set(df.columns[df.apply(lambda x: len(x.unique()), axis = 0)/df.count()*100 < uniqueCut])
        HighFreq = set(df.columns[df.apply(lambda x: x.value_counts().tolist()[0]/x.value_counts().tolist()[1]) > freqCut])
        return df.drop(LowCut & HighFreq, axis = 1)
    
    else:
        isinstance(df, pd.DataFrame)
        df = df.drop(df.columns[df.apply(lambda x: len(x.unique()), axis = 0) <2], axis = 1)
        LowCut = set(df.columns[df.apply(lambda x: len(x.unique()), axis = 0)/df.count()*100 < uniqueCut])
        HighFreq = set(df.columns[df.apply(lambda x: x.value_counts().tolist()[0]/x.value_counts().tolist()[1]) > freqCut])
        return df.drop(LowCut & HighFreq, axis = 1)
    

# Find Corralated
def find_correlation(df, thresh=0.9):
    """
    Stolen from R's caret: Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with
               a correlation greater than this value
    """
    
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)


    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Code stolen from scikit-learn user guide. Generate a simple plot 
    of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt