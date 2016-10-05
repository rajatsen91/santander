#Author : Rajat Sen
#HW0 - LSML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import ctypes
import copy
import sys
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix, hstack, csr_matrix, vstack
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier
from pyfm import pylibfm
# from adaboost_multiple import AdaBoost
from itertools import combinations
from sklearn import metrics, cross_validation, linear_model
# from logistic_regression_updated import group_data,OneHotEncoder2
from sklearn.cross_validation import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import itertools
# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)


def load_data(filetrain,filetest):
    '''Function to load train and test data into pandas data frame. 
    Argument1 : training dataset filename
    Argument2 : test dataset filename
    '''
    train_df = pd.read_csv(filetrain, header=0)
    test_df = pd.read_csv(filetest, header=0)
    
    return train_df, test_df


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def convert_to_category(A):
    (m,n) = A.shape

    for j in range(n):
        count = 0
        dic = {}
        for i in range(m):
            if A[i,j] in dic:
                A[i,j] = dic[A[i,j]]
            else:
                count = count + 1
                dic[A[i,j]] = count
                A[i,j] = dic[A[i,j]]

    return A





def preprocess(train_df,test_df):
    col_test = 'TARGET' 

    print 'in preprocessing'
    
    big_X = train_df.drop(['ID', 'TARGET'], axis=1).append(test_df.drop(['ID'], axis=1))
    big_X_imputed = DataFrameImputer().fit_transform(big_X)
    train_y = train_df[col_test]

    train_X = csr_matrix(big_X_imputed[0:train_df.shape[0]].as_matrix())
    test_X = csr_matrix(big_X_imputed[train_df.shape[0]::].as_matrix())

    print train_X.shape
    print test_X.shape

    indices = []
    for i in range(train_X.shape[1]):
        x = np.count_nonzero(train_X[:,i].todense().T)
        y = set(train_X[:,1].toarray().T[0,:])
        if x != 0 and len(y) > 1:
            indices = indices + [i]

    train_X_ncol = train_X[:,indices]
    test_X_ncol = test_X[:,indices]

    print train_X_ncol.shape
    print test_X_ncol.shape

    big_X_ncol = vstack([train_X_ncol,test_X_ncol])

    big_X_freq = csr_matrix(freq_accuracy(big_X_ncol.todense(),train_y))

    train_X_freq1 = big_X_freq[0:train_df.shape[0]]
    test_X_freq1 = big_X_freq[train_df.shape[0]::]

    train_X_freq = hstack([train_X_ncol,train_X_freq1])
    test_X_freq = hstack([test_X_ncol,test_X_freq1])

    print train_X_freq.shape
    print test_X_freq.shape

   





    
    save_sparse_csr('train_X', train_X)
    save_sparse_csr('test_X', test_X)

    save_sparse_csr('train_X_ncol', train_X_ncol) 
    save_sparse_csr('test_X_ncol', test_X_ncol)

    save_sparse_csr('train_X_freq', csr_matrix(train_X_freq)) 
    save_sparse_csr('test_X_freq', csr_matrix(test_X_freq))


def new_features(train_df,test_df):
    train_X_freq = load_sparse_csr('train_X_freq.npz')
    test_X_freq = load_sparse_csr('test_X_freq.npz')

    train_X_ncol = load_sparse_csr('train_X_ncol.npz')
    test_X_ncol = load_sparse_csr('test_X_ncol.npz')

    pca = PCA(n_components=2)
    x_train_projected = pca.fit_transform(normalize(train_X_freq, axis=0).toarray())
    x_test_projected = pca.transform(normalize(test_X_freq, axis=0).toarray())

    x_train_nnz = np.zeros([train_X_freq.shape[0],1])
    x_test_nnz = np.zeros([test_X_freq.shape[0],1])

    for i in range(train_X_freq.shape[0]):
        x_train_nnz[i,0] = np.count_nonzero(train_X_freq[i,:].todense())

    for i in range(test_X_freq.shape[0]):
        x_test_nnz[i,0] = np.count_nonzero(test_X_freq[i,:].todense())

    print train_X_freq.shape
    print x_train_nnz.shape
    print x_train_projected.shape
    train_X_pca = hstack([train_X_freq,x_train_projected,x_train_nnz])
    test_X_pca = hstack([test_X_freq,x_test_projected,x_test_nnz])
    postrain = 0

    var15tr = train_df['var15']
    saldo_medio_var5_hace2tr = train_df['saldo_medio_var5_hace2']
    saldo_var33tr = train_df['saldo_var33']
    var38tr = train_df['var38']
    V21tr = train_df['var21']

    var15 = test_df['var15']
    saldo_medio_var5_hace2 = test_df['saldo_medio_var5_hace2']
    saldo_var33 = test_df['saldo_var33']
    var38 = test_df['var38']
    V21 = test_df['var21']

    train_X_pca = train_X_pca.todense()
    test_X_pca = test_X_pca.todense()

    



    for i in range(train_df.shape[0]):
        if var15tr[i] < 23 :
            postrain = i
            break

    for i in range(train_df.shape[0]):
        if var15tr[i] < 23 or saldo_medio_var5_hace2tr[i] > 160000 or saldo_var33tr[i] > 0 or var38tr[i] > 3988596 or V21tr[i]>7500 :
            train_X_pca[i,:] = train_X_pca[postrain,:]

    for i in range(test_df.shape[0]):
        if var15[i] < 23 or saldo_medio_var5_hace2[i] > 160000 or saldo_var33[i] > 0 or var38[i] > 3988596 or V21[i]>7500 :
            test_X_pca[i,:] = train_X_pca[postrain,:]







    save_sparse_csr('train_X_pca_mod', csr_matrix(train_X_pca)) 
    save_sparse_csr('test_X_pca_mod', csr_matrix(test_X_pca))

    # ncol = train_X_ncol.shape[1]
    # train_freq = train_X_freq[:,ncol::]
    # test_freq = test_X_freq[:,ncol::]

    # train_X_product = copy.deepcopy(train_X_pca)
    # test_X_product = copy.deepcopy(test_X_pca)
    # count = 0

    # for subset in itertools.combinations(range(train_freq.shape[1]), 2):
    #     s = list(subset)
    #     ltrain = np.multiply(train_freq[:,s[0]].todense(),train_freq[:,s[1]].todense())
    #     ltest = np.multiply(test_freq[:,s[0]].todense(),test_freq[:,s[1]].todense())
    #     train_X_product = hstack([train_X_product,ltrain])
    #     test_X_product = hstack([test_X_product,ltest])
    #     count = count + 1
    #     if count%50 == 0:
    #         print count
    #         print s


    # save_sparse_csr('train_X_product', csr_matrix(train_X_product)) 
    # save_sparse_csr('test_X_product', csr_matrix(test_X_product))










    


def cross_validate(classifier, n_folds = 5):
    train_X = classifier['train_X']
    train_y = classifier['train_y']
    model = classifier['model']
    score = 0.0
    
    skf = StratifiedKFold(train_y, n_folds=n_folds)
    for train_index, test_index in skf:
        X_train, X_test = train_X[train_index], train_X[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        clf = model.fit(X_train,y_train)
        pred = clf.predict_proba(X_test)[:,1]
        print 'cross', roc_auc_score(y_test,pred)
        score = score + roc_auc_score(y_test,pred)

    return score/n_folds


def general_stackout(classifiers):
    train_X = classifiers[0]['train_X'] 
    train_y = classifiers[0]['train_y']
    test_X = classifiers[0]['test_X']
    model = classifiers[0]['model']


    num_train = int(0.5*train_X.shape[0])
    print num_train
    train_X1 = train_X[0:num_train]
    train_X2 = train_X[num_train::]
    train_y1 = train_y[0:num_train]
    train_y2 = train_y[num_train::]

    m1 = model.fit(train_X1,train_y1)
    m2 = model.fit(train_X2,train_y2)
    m = model.fit(train_X,train_y)

    pred2 = np.transpose(np.array([m1.predict_proba(train_X2)[:,1]]))
    pred1 = np.transpose(np.array([m2.predict_proba(train_X1)[:,1]]))

    print pred2.shape
    print pred1.shape

    pred = np.vstack((pred1,pred2))

    print pred.shape

    pred_test = np.transpose(np.array([m.predict_proba(test_X)[:,1]]))

    pred_train = copy.deepcopy(pred)

    for i in range(1,len(classifiers)):
        train_X = classifiers[i]['train_X'] 
        train_y = classifiers[i]['train_y']
        test_X = classifiers[i]['test_X']
        model = classifiers[i]['model']


        num_train = int(0.5*train_X.shape[0])
        train_X1 = train_X[0:num_train]
        train_X2 = train_X[num_train::]
        train_y1 = train_y[0:num_train]
        train_y2 = train_y[num_train::]

        m1 = model.fit(train_X1,train_y1)
        m2 = model.fit(train_X2,train_y2)
        m = model.fit(train_X,train_y)

        pred2 = np.transpose(np.array([m1.predict_proba(train_X2)[:,1]]))
        pred1 = np.transpose(np.array([m2.predict_proba(train_X1)[:,1]]))

        pred = np.vstack((pred1,pred2))

        pred_train = np.hstack((pred_train,pred))

        pred_t = np.transpose(np.array([m.predict_proba(test_X)[:,1]]))

        pred_test = np.hstack((pred_test,pred_t))



    return pred_train, pred_test


def inverse_logistic(A):
    print A.shape
    B = copy.deepcopy(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            try:
                B[i,j] = math.log(A[i,j]/(1 - A[i,j]))
            except:
                B[i,j] = -1e20

    return B



def ensemble_predict_cv(train_df,test_df):
    col_categorical = ['var2','var3','var7','var10','var13','var14','var15','var21']
    col_normal = ['var1' , 'var4', 'var5' ,'var6' , 'var8' , 'var9' ,'var11', 'var12' ,'var16', 'var17', 'var18', 'var19', 'var20' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = csr_matrix(np.load('train_X_whole_onebit.npy'))
    test_X_whole_onebit = csr_matrix(np.load('test_X_whole_onebit.npy'))

    train_X_whole_cat_hd = load_sparse_csr('Ntrain_X_whole_cat_hd.npz')
    test_X_whole_cat_hd = load_sparse_csr('Ntest_X_whole_cat_hd.npz')


    train_X_whole_freq = np.load('Ntrain_X_whole_freq.npy')
    test_X_whole_freq = np.load('Ntest_X_whole_freq.npy')

    train_X_whole_all = np.load('train_X_whole_all.npy')
    test_X_whole_all = np.load('test_X_whole_all.npy')


    classifiers = [{}]*3

    classifiers[0]['model'] = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=4,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.4,objective= 'binary:logistic',scale_pos_weight=1, seed=11)
    classifiers[0]['train_X'] = train_X_whole_freq
    classifiers[0]['train_y'] = train_y
    classifiers[0]['test_X'] = test_X_whole_freq


    classifiers[1]['model'] = RandomForestClassifier(random_state = 11, n_estimators = 300, max_depth = 10, criterion = 'entropy')
    classifiers[1]['train_X'] = train_X_whole_freq
    classifiers[1]['train_y'] = train_y
    classifiers[1]['test_X'] = test_X_whole_freq


    classifiers[2]['model'] = LogisticRegression(penalty = 'l1', random_state = 11, C = 0.1)
    classifiers[2]['train_X'] = train_X_whole_cat_hd
    classifiers[2]['train_y'] = train_y
    classifiers[2]['test_X'] = test_X_whole_cat_hd


    train_ensemble, test_ensemble = general_stackout(classifiers)

    train_ensemble_inv = np.hstack((train_X_whole_all, inverse_logistic(train_ensemble)))
    test_ensemble_inv = np.hstack((test_X_whole_all, inverse_logistic(test_ensemble)))


    print 'ensmebling'
    param_test1 = {'max_depth':[3,4,6],'n_estimators' : [200,300], 'colsample_bytree' : [0.4,0.8]}
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1, seed=11), param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(train_ensemble_inv,train_y)
    bp = gsearch1.best_params_
    print bp
    
    gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11).fit(train_ensemble_inv, train_y)
    predictions = gbm.predict_proba(test_ensemble_inv)[:,1]
    
    classifier = {}
    classifier['model'] = xgb.XGBClassifier( learning_rate =0.05, n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=bp['colsample_bytree'],objective= 'binary:logistic',scale_pos_weight=1, seed=11)
    classifier['train_X'] = train_ensemble_inv
    classifier['train_y'] = train_y

    print cross_validate(classifier,2,10)


    return predictions



















    
  

def freq_data(big_X):
    X = np.matrix(big_X)
    (m,n) = X.shape
    for j in range(n):
        dic = {}
        for i in range(m):
            if X[i,j] in dic:
                dic[X[i,j]] = dic[X[i,j]] + 1
            else:
                dic[X[i,j]] = 0

        for i in range(m):
            X[i,j] = dic[X[i,j]]

    return X


def freq_accuracy(big_X,train_y):
    X = np.matrix(big_X)
    (m,n) = X.shape
    t_n = len(train_y)
    Y = np.zeros((m,2*n))
    for j in range(n):
        dic1 = {}
        dic2 = {}
        for i in range(m):
            if X[i,j] in dic1:
                dic1[X[i,j]] = dic1[X[i,j]] + 1
            else:
                dic1[X[i,j]] = 1

            if j < t_n:
                if X[i,j] in dic2:
                    dic2[X[i,j]] = dic2[X[i,j]] + train_y[j]
                else:
                    dic2[X[i,j]] = train_y[j]

        for i in range(m):
            if X[i,j] != 0:
                Y[i,j] = dic1[X[i,j]]
                Y[i,j+n] = float(1.0*dic2[X[i,j]]/dic1[X[i,j]])
            else:
                Y[i,j] = 0
                Y[i,j+n] = 0

    return Y



    
def xgb_boost_predict_cv(train_df,test_df):
    '''function that corss-validates over certain parameters and predicts with the best parameter set'''
    # col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    # col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    # col_normal = [ 'var4', 'var12' , 'var18' ]
    col_test = 'TARGET' 
    
    train_y = train_df[col_test]

    train = load_sparse_csr('train_X_pca_mod.npz')
    test = load_sparse_csr('test_X_pca_mod.npz')






    print 'training xgb'

    
    gbm = xgb.XGBClassifier( learning_rate =0.0202048, n_estimators=560, max_depth=5 ,min_child_weight=1, gamma=0.1, subsample=0.6815, colsample_bytree=0.701 ,objective= 'binary:logistic',scale_pos_weight=1, seed=1234).fit(train, train_y)
    indices = np.nonzero(gbm.feature_importances_)[0]

    print indices
    train = train[:,indices]
    test = test[:,indices]

    # gbm = xgb.XGBClassifier( learning_rate =0.0202048, n_estimators=560, max_depth=5 ,min_child_weight=1, gamma=0.1, subsample=0.6815, colsample_bytree=0.701 ,objective= 'binary:logistic',scale_pos_weight=1, seed=1234).fit(train, train_y)
    # predictions = gbm.predict_proba(test)[:,1]

    classifier = {}
    classifier['model'] =xgb.XGBClassifier( learning_rate =0.0202048, n_estimators=560, max_depth=5 ,min_child_weight=1, gamma=0.1, subsample=0.6815, colsample_bytree=0.701 ,objective= 'binary:logistic',scale_pos_weight=1, seed=1234)
    classifier['train_X'] = train
    classifier['train_y'] = train_y

    print 'cross-validating'

    print cross_validate(classifier,5)
    
    return predictions


def pyfm_predict_cv(train_df,test_df):
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_normal = [ 'var4', 'var12' , 'var18' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = csr_matrix(np.load('train_X_whole_onebit.npy'))
    test_X_whole_onebit = csr_matrix(np.load('test_X_whole_onebit.npy'))

    train_X_whole_cat_hd = csr_matrix(np.load('train_X_whole_cat_hd.npy'))
    test_X_whole_cat_hd = csr_matrix(np.load('test_X_whole_cat_hd.npy'))


    train_X_whole_freq = np.load('train_X_whole_freq.npy')
    test_X_whole_freq = np.load('test_X_whole_freq.npy')

    train_X_whole_all = np.load('train_X_whole_all.npy')
    test_X_whole_all = np.load('test_X_whole_all.npy')

    skf = StratifiedKFold(train_y, n_folds=2)
    for train_index, test_index in skf:
        X_train, X_test = train_X_whole_cat_hd[train_index], train_X_whole_cat_hd[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
    fm = pylibfm.FM(num_factors=70, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.001, learning_rate_schedule="optimal")
    fm.fit(X_train, y_train.astype(np.float))
    predictions = fm.predict(X_test)
    print roc_auc_score(y_test,predictions)
    


    
    
def logistic_predict_cv(train_df,test_df):
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_normal = [ 'var4', 'var12' , 'var18' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = load_sparse_csr('Ntrain_X_whole_onebit.npz')
    test_X_whole_onebit = load_sparse_csr('Ntest_X_whole_onebit.npz')

    train_X_whole_cat_hd = load_sparse_csr('Ntrain_X_whole_cat_hd.npz')
    test_X_whole_cat_hd = load_sparse_csr('Ntest_X_whole_cat_hd.npz')


    train_X_whole_freq = np.load('Ntrain_X_whole_freq.npy')
    test_X_whole_freq = np.load('Ntest_X_whole_freq.npy')

    train_X_whole_all = load_sparse_csr('Ntrain_X_whole_all.npz')
    test_X_whole_all = load_sparse_csr('Ntest_X_whole_all.npz')

    train_X_onebit_categorical = load_sparse_csr('train_X_onebit_categorical.npz')
    test_X_onebit_categorical = load_sparse_csr('test_X_onebit_categorical.npz')

    train_X_normal= np.load('train_X_normal.npy')
    test_X_normal= np.load('test_X_normal.npy')



    param_test1 = {'C':[1e-2,0.1,0.5,1,1.6,1.8,2],'penalty' : ['l1','l2']}
    gsearch1 = GridSearchCV(estimator = LogisticRegression(penalty = 'l1', random_state = 11, C = 1) , param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
    gsearch1.fit(train_X_normal,train_y)
    bp = gsearch1.best_params_
    print bp
    logit = LogisticRegression(penalty = bp['penalty'], random_state = 11, C = bp['C']).fit(train_X_normal,train_y)
    predictions = logit.predict_proba(test_X_normal)[:,1]
    
    classifier = {}
    classifier['model'] = LogisticRegression(penalty = bp['penalty'], random_state = 11, C = bp['C'])
    classifier['train_X'] = train_X_normal
    classifier['train_y'] = train_y

    print cross_validate(classifier,2,5)
    


    return predictions



def extratreesboost_predict_cv(train_df,test_df):
    col_categorical = ['var2','var3','var7','var10','var13','var14','var15','var21']
    col_normal = ['var1' , 'var4', 'var5' ,'var6' , 'var8' , 'var9' ,'var11', 'var12' ,'var16', 'var17', 'var18', 'var19', 'var20' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = csr_matrix(np.load('train_X_whole_onebit.npy'))
    test_X_whole_onebit = csr_matrix(np.load('test_X_whole_onebit.npy'))

    train_X_whole_cat_hd = csr_matrix(np.load('train_X_whole_cat_hd.npy'))
    test_X_whole_cat_hd = csr_matrix(np.load('test_X_whole_cat_hd.npy'))


    train_X_whole_freq = np.load('train_X_whole_freq.npy')
    test_X_whole_freq = np.load('test_X_whole_freq.npy')


    # param_test1 = {'n_estimators':[200,300,400],'criterion' : ['gini','entropy'], 'bootstrap' : [True, False]}
    # gsearch1 = GridSearchCV(estimator = ExtraTreesClassifier(criterion = 'gini',random_state = 11, n_estimators = 20, bootstrap = False) , param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    # gsearch1.fit(train_X_freq,train_y)
    # bp = gsearch1.best_params_
    # print bp
    adb = AdaBoostClassifier(n_estimators = 70, base_estimator = ExtraTreesClassifier(criterion = 'entropy',random_state = 11, n_estimators = 300, bootstrap = True)).fit(train_X_whole_freq,train_y)
    predictions = adb.predict_proba(test_X_whole_freq)[:,1]


    classifier = {}
    classifier['model'] = AdaBoostClassifier(n_estimators = 150, base_estimator = ExtraTreesClassifier(criterion = 'entropy',random_state = 11, n_estimators = 300, bootstrap = True))
    classifier['train_X'] = train_X_whole_freq
    classifier['train_y'] = train_y

    print cross_validate(classifier,2,5)
    
    return predictions


def extratrees_predict_cv(train_df,test_df):
    col_test = 'TARGET' 
    
    train_y = train_df[col_test]

    train = load_sparse_csr('train_X_freq.npz')
    test = load_sparse_csr('test_X_freq.npz')






    print 'training ET'
    
    ET = ExtraTreesClassifier(criterion = 'entropy',random_state = 11, n_estimators = 400, bootstrap = True).fit(train,train_y)
    indices = np.nonzero(ET.feature_importances_)[0]

    print ET.feature_importances_

    print 'Training ET'

    train = train[:,indices]
    test = test[:,indices]


    # param_test1 = {'n_estimators':[200,300,400],'criterion' : ['entropy'], 'bootstrap' : [True]}
    # gsearch1 = GridSearchCV(estimator = ExtraTreesClassifier(criterion = 'gini',random_state = 11, n_estimators = 20, bootstrap = False) , param_grid = param_test1, scoring='roc_auc',n_jobs=5,iid=False, cv=5)
    # gsearch1.fit(train,train_y)
    # bp = gsearch1.best_params_
    # print bp
    ET = ExtraTreesClassifier(criterion = 'entropy',random_state = 11, n_estimators = 400, bootstrap = True).fit(train,train_y)
    predictions = ET.predict_proba(test)[:,1]

    # print ET.feature_importances_

    classifier = {}
    print 'cross_validate'
    classifier['model'] = ExtraTreesClassifier(criterion = 'entropy',random_state = 11, n_estimators = 400, bootstrap = True)
    classifier['train_X'] = train
    classifier['train_y'] = train_y

    print cross_validate(classifier,5)
    
    return predictions


def randomforrest_predict_cv(train_df,test_df):
    col_test = 'TARGET' 
    
    train_y = train_df[col_test]

    train = load_sparse_csr('train_X_pca.npz')
    test = load_sparse_csr('test_X_pca.npz')

    clf = RandomForestClassifier(random_state = 11, n_estimators = 300, max_depth = 10, criterion = 'entropy').fit(train,train_y)
    indices = np.nonzero(clf.feature_importances_)[0]

    print 'Training RF'

    train = train[:,indices]
    test = test[:,indices]

    # np.save('train_RF',train)
    # np.save('test_RF',test)

    # param_test1 = {'n_estimators':[200,300,400], 'max_depth' : [5,10,None], 'criterion' : ['entropy']}
    # gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state = 11, n_estimators = 10, max_depth = None, criterion = 'gini') , param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    # gsearch1.fit(train,train_y)
    # bp = gsearch1.best_params_
    # print bp
    # RandomForestClassifier(random_state = 11, n_estimators = bp['n_estimators'], max_depth = bp['max_depth'], criterion = bp['criterion'])
    clf = RandomForestClassifier(random_state = 11, n_estimators = 300, max_depth = 10, criterion = 'entropy').fit(train,train_y)
    predictions =clf.predict_proba(test)[:,1]

    
    print 'Cross Validating'

    classifier = {}
    classifier['model'] = RandomForestClassifier(random_state = 11, n_estimators = 300, max_depth = 10, criterion = 'entropy')
    classifier['train_X'] = train
    classifier['train_y'] = train_y
    print cross_validate(classifier,5)

    return predictions





def RFboost_predict_cv(train_df,test_df):
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_normal = [ 'var4', 'var12' , 'var18' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = load_sparse_csr('Ntrain_X_whole_onebit.npz')
    test_X_whole_onebit = load_sparse_csr('Ntest_X_whole_onebit.npz')

    train_X_whole_cat_hd = load_sparse_csr('Ntrain_X_whole_cat_hd.npz')
    test_X_whole_cat_hd = load_sparse_csr('Ntest_X_whole_cat_hd.npz')


    train_X_whole_freq = np.load('Ntrain_X_whole_freq.npy')
    test_X_whole_freq = np.load('Ntest_X_whole_freq.npy')

    train_X_whole_all = load_sparse_csr('Ntrain_X_whole_all.npz')
    test_X_whole_all = load_sparse_csr('Ntest_X_whole_all.npz')
    # param_test1 = {'n_estimators':[200,300,400],'criterion' : ['gini','entropy'], 'bootstrap' : [True, False]}
    # gsearch1 = GridSearchCV(estimator = ExtraTreesClassifier(criterion = 'gini',random_state = 11, n_estimators = 20, bootstrap = False) , param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    # gsearch1.fit(train_X_freq,train_y)
    # bp = gsearch1.best_params_
    # print bp
    adb = AdaBoostClassifier(n_estimators = 70, base_estimator = RandomForestClassifier(random_state = 11, n_estimators = 400, max_depth = 10, criterion = 'entropy')).fit(train_X_whole_freq,train_y)
    predictions = adb.predict_proba(test_X_whole_freq)[:,1]

    classifier = {}
    classifier['model'] = AdaBoostClassifier(n_estimators = 70, base_estimator = RandomForestClassifier(random_state = 11, n_estimators = 400, max_depth = 10, criterion = 'entropy'))
    classifier['train_X'] = train_X_whole_freq
    classifier['train_y'] = train_y
    print cross_validate(classifier,2,5)

    return predictions


def SVC_predict_cv(train_df,test_df):
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_normal = [ 'var4', 'var12' , 'var18' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = load_sparse_csr('Ntrain_X_whole_onebit.npz')
    test_X_whole_onebit = load_sparse_csr('Ntest_X_whole_onebit.npz')

    train_X_whole_cat_hd = load_sparse_csr('Ntrain_X_whole_cat_hd.npz')
    test_X_whole_cat_hd = load_sparse_csr('Ntest_X_whole_cat_hd.npz')


    train_X_whole_freq = np.load('Ntrain_X_whole_freq.npy')
    test_X_whole_freq = np.load('Ntest_X_whole_freq.npy')

    train_X_whole_all = load_sparse_csr('Ntrain_X_whole_all.npz')
    test_X_whole_all = load_sparse_csr('Ntest_X_whole_all.npz')


    param_test1 = {'C':[1e-1,1,3,10],'kernel':['rbf','poly','sigmoid']}
    gsearch1 = GridSearchCV(estimator = SVC(C = 1.0,random_state = 11, kernel = 'rbf', probability = True) , param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(train_X_whole_freq,train_y)
    bp = gsearch1.best_params_
    print bp
    logit = SVC(C = bp['C'],random_state = 11, kernel = bp['kernel'], probability = True).fit(train_X_whole_freq,train_y)
    predictions = logit.predict_proba(test_X_whole_freq)[:,1]

    classifier = {}
    classifier['model'] =  SVC(C = bp['C'],random_state = 11, kernel = bp['kernel'], probability = True)
    classifier['train_X'] = train_X_whole_freq
    classifier['train_y'] = train_y
    print cross_validate(classifier,2,5)
    
    return predictions


def linearSVC_predict_cv(train_df,test_df):
    print 'here'
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_categorical = ['var1','var2','var3','var5' ,'var6','var7','var8' , 'var9','var10','var11','var13','var14','var15','var16', 'var17', 'var19', 'var20', 'var21']
    col_normal = [ 'var4', 'var12' , 'var18' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = load_sparse_csr('Ntrain_X_whole_onebit.npz')
    test_X_whole_onebit = load_sparse_csr('Ntest_X_whole_onebit.npz')

    train_X_whole_cat_hd = load_sparse_csr('Ntrain_X_whole_cat_hd.npz')
    test_X_whole_cat_hd = load_sparse_csr('Ntest_X_whole_cat_hd.npz')


    train_X_whole_freq = np.load('Ntrain_X_whole_freq.npy')
    test_X_whole_freq = np.load('Ntest_X_whole_freq.npy')

    train_X_whole_all = load_sparse_csr('Ntrain_X_whole_all.npz')
    test_X_whole_all = load_sparse_csr('Ntest_X_whole_all.npz')

    param_test1 = {'C':[1,1.2,1.4,1.6,2,3,3.5]}
    gsearch1 = GridSearchCV(estimator = LinearSVC(C = 1.0,random_state = 11, penalty = 'l1',dual = False) , param_grid = param_test1, scoring='roc_auc',n_jobs=20,iid=False, cv=5)
    gsearch1.fit(train_X_whole_freq,train_y)
    bp = gsearch1.best_params_
    print bp
    logit = LinearSVC(C = bp['C'],random_state = 11, penalty = 'l1', dual = False).fit(train_X_whole_freq, train_y)
    predictions = logit.predict(test_X_whole_freq)

    classifier = {}
    classifier['model'] =  LinearSVC(C = bp['C'],random_state = 11, penalty = 'l1', dual = False)
    classifier['train_X'] = train_X_whole_freq
    classifier['train_y'] = train_y
    print cross_validate(classifier,2,5)
    
    return predictions


def SGD_boost(train_df,test_df):
    col_categorical = ['var2','var3','var7','var10','var13','var14','var15','var21']
    col_normal = ['var1' , 'var4', 'var5' ,'var6' , 'var8' , 'var9' ,'var11', 'var12' ,'var16', 'var17', 'var18', 'var19', 'var20' ]
    col_test = 'response' 
    col_to_use = col_categorical + col_normal
    
    train_y = train_df[col_test]

    train_X_whole_onebit = csr_matrix(np.load('train_X_whole_onebit.npy'))
    test_X_whole_onebit = csr_matrix(np.load('test_X_whole_onebit.npy'))

    train_X_whole_cat_hd = csr_matrix(np.load('train_X_whole_cat_hd.npy'))
    test_X_whole_cat_hd = csr_matrix(np.load('test_X_whole_cat_hd.npy'))


    train_X_whole_freq = np.load('train_X_whole_freq.npy')
    test_X_whole_freq = np.load('test_X_whole_freq.npy')

    param_test1 = {'alpha':[1e-2,0.1,0.5,1,2], 'l1_ratio' : [0,0.1,0.2,0.3,0.6]}
    gsearch1 = GridSearchCV(estimator = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = 0.1, l1_ratio = 0.95) , param_grid = param_test1, scoring='roc_auc',n_jobs=5,iid=False, cv=5)
    gsearch1.fit(train_X_whole_cat_hd,train_y)
    bp = gsearch1.best_params_
    print bp

    SGD = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = bp['alpha'], l1_ratio = bp['l1_ratio']).fit(train_X_whole_cat_hd,train_y)
    predictions = SGD.predict(test_X_whole_cat_hd)

    classifier = {}
    classifier['model'] = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = bp['alpha'], l1_ratio = bp['l1_ratio'])
    classifier['train_X'] = train_X_whole_cat_hd
    classifier['train_y'] = train_y
    print cross_validate(classifier,2,5)

    adb = AdaBoostClassifier(n_estimators = 70, base_estimator = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = bp['alpha'], l1_ratio = bp['l1_ratio'])).fit(train_X_whole_cat_hd,train_y)
    predictions = adb.predict_proba(test_X_whole_cat_hd)[:,1]

    classifier = {}
    classifier['model'] = AdaBoostClassifier(n_estimators = 10, base_estimator = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = bp['alpha'], l1_ratio = bp['l1_ratio']))
    classifier['train_X'] = train_X_whole_cat_hd
    classifier['train_y'] = train_y
    print cross_validate(classifier,2,5)



    return predictions

def sigmoid(x):

    for i in range(x.shape[0]):
        try:
            x[i,0] = 1 / (1 + math.exp(-x[i,0])) 
        except:
            if x[i,0] < 0:
                x[i,0] = 1
            else:
                x[i,0] = 0

    return x

def rank(array):
    temp = array.argsort(axis = 0)
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))

    return ranks

def holdout(train_df,test_df):
    col_test = 'TARGET' 
    
    train_y = train_df[col_test]

    train = load_sparse_csr('train_X_pca_mod.npz')
    test = load_sparse_csr('test_X_pca_mod.npz')
    print 'Selecting Features'
    gbm = xgb.XGBClassifier( learning_rate =0.05, n_estimators=180, max_depth=5 ,min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.4,objective= 'binary:logistic',scale_pos_weight=1, seed = 11).fit(train,train_y)
    indices = np.nonzero(gbm.feature_importances_)[0]

    print indices
    train = train[:,indices]
    test = test[:,indices]


    hf = int(train_df.shape[0]*0.7)


    train_t = train[0:hf]
    train_h = train[hf::]


    train_y_t = train_y[0:hf]
    train_y_h = train_y[hf::]
    # print 'training logit'
    # logit = LogisticRegression(penalty = 'l1', random_state = 11, C = 0.1).fit(train_X_whole_cat_hd_t,train_y_t)
    # print 'training ET'
    # ET = ExtraTreesClassifier(criterion = 'entropy',random_state = 11, n_estimators = 400, bootstrap = True).fit(train_X_whole_freq_t,train_y_t)
    print 'training xgb'
    gbm = xgb.XGBClassifier( learning_rate =0.0202048, n_estimators=560, max_depth=5 ,min_child_weight=1, gamma=0.1, subsample=0.6815, colsample_bytree=0.701 ,objective= 'binary:logistic',scale_pos_weight=1, seed=1234).fit(train_t,train_y_t)
    print 'training RF'
    RF = RandomForestClassifier(random_state = 11, n_estimators = 300, max_depth = 10, criterion = 'entropy').fit(train_t,train_y_t)

    # predlog = rank(np.array([logit.predict_proba(train_X_whole_cat_hd_h)[:,1]]).T)
    # predET= rank(np.array([ET.predict_proba(train_X_whole_freq_h)[:,1]]).T)
    predgbm = rank(np.array([gbm.predict_proba(train_h)[:,1]]).T)
    predRF = rank(np.array([RF.predict_proba(train_h)[:,1]]).T)

    # print predRF


    
    
    a0 = 0
    a1 = 0
    a2 = 8
    a3 = 2

    pred = (a2*predgbm + a3*predRF)/(a2 + a3)


    var15tr = train_df['var15']
    saldo_medio_var5_hace2tr = train_df['saldo_medio_var5_hace2']
    saldo_var33tr = train_df['saldo_var33']
    var38tr = train_df['var38']
    V21tr = train_df['var21']

    for i in range(len(pred)):
        if var15tr[i] < 23 or saldo_medio_var5_hace2tr[i] > 160000 or saldo_var33tr[i] > 0 or var38tr[i] > 3988596 or V21tr[i]>7500 :
            pred[i] = 0

    print roc_auc_score(train_y_h,pred)

    # logitf = LogisticRegression(penalty = 'l1', random_state = 11, C = 0.1).fit(train_X_whole_cat_hd,train_y)
    # ETf = ExtraTreesClassifier(criterion = 'entropy',random_state = 11, n_estimators = 400, bootstrap = True).fit(train_X_whole_freq,train_y)
    gbmf = xgb.XGBClassifier( learning_rate =0.0202048, n_estimators=560, max_depth=5 ,min_child_weight=1, gamma=0.1, subsample=0.6815, colsample_bytree=0.701 ,objective= 'binary:logistic',scale_pos_weight=1, seed=1234).fit(train,train_y)
    RFf = RandomForestClassifier(random_state = 11, n_estimators = 300, max_depth = 10, criterion = 'entropy').fit(train,train_y)

    # predlogf = rank(np.array([logitf.predict_proba(test_X_whole_cat_hd)[:,1]]).T)
    # predETf = rank(np.array([ETf.predict_proba(test_X_whole_freq)[:,1]]).T)
    predgbmf = rank(np.array([gbmf.predict_proba(test)[:,1]]).T)
    predRFf = rank(np.array([RFf.predict_proba(test)[:,1]]).T)

    predf = (a2*predgbmf + a3*predRFf)/( a2 + a3)

    var15 = test_df['var15']
    saldo_medio_var5_hace2 = test_df['saldo_medio_var5_hace2']
    saldo_var33 = test_df['saldo_var33']
    var38 = test_df['var38']
    V21 = test_df['var21']

    for i in range(len(predf)):
        if var15[i] < 23 or saldo_medio_var5_hace2[i] > 160000 or saldo_var33[i] > 0 or var38[i] > 3988596 or V21[i]>7500 :
            predf[i] = 0
    print predf.shape
    predf = predf.astype(float)
    predictions = (predf/np.max(predf))
    return predictions










    


def write_to_file(test_df,predictions):
    
    submission = pd.DataFrame({ 'ID': test_df['ID'],'TARGET': predictions })
    submission = submission[['ID','TARGET']]
    submission.to_csv("submission_zeroed.csv", index=False)
    
def write_test_hack():
    x1 = pd.read_csv('submission(1).csv', header=0)
    x2 = pd.read_csv('submission_2.csv', header=0)
    x3 = pd.read_csv('submission_xgb_gridsearchcv2.csv', header=0)
    # # x3 = pd.read_csv('submission_new_xgb83498.csv', header=0)
    # # x4 = pd.read_csv('submission_RF.csv', header=0)
    
    a1 = 80 #1 - 0.63
    a2 = 10 #0.63
    a3 = 10
    # # a4 = 15

    # # x1 = pd.read_csv('submission.csv', header=0)
    # # pred1 = np.array(x1['TARGET']).astype(float)





    pred1 = rank(np.array(x1['TARGET'])).astype(float)
    pred2 = rank(np.array(x2['TARGET'])).astype(float)
    pred3 = rank(np.array(x3['TARGET'])).astype(float)
    # # pred4 = rank(np.array(x4['response']))

    preds = (a1*pred1 + a2*pred2 + a3*pred3)/(a1 + a2+a3)
    preds = preds/np.amax(preds)

    # x = pd.read_csv('submission_1.csv', header=0)
    # preds = np.array(x['TARGET']).astype(float)
    var15 = test_df['var15']
    saldo_medio_var5_hace2 = test_df['saldo_medio_var5_hace2']
    saldo_var33 = test_df['saldo_var33']
    var38 = test_df['var38']
    v21 = test_df['var21']
    nv = test_df['num_var33']+test_df['saldo_medio_var33_ult3']+test_df['saldo_medio_var44_hace2']+test_df['saldo_medio_var44_hace3']+ test_df['saldo_medio_var33_ult1']+test_df['saldo_medio_var44_ult1']
    num_var30 = test_df['num_var30']
    num_var13_0 = test_df['num_var13_0']
    num_var33_0 = test_df['num_var33_0']
    x0 = test_df['imp_ent_var16_ult1'] #> 51003 = 0
    x1 = test_df['imp_op_var39_comer_ult3'] #> 13184] = 0
    x2 = test_df['saldo_medio_var5_ult3'] #> 108251] = 0
    x3 = test_df['var15']+test_df['num_var45_hace3']+test_df['num_var45_ult3']+test_df['var36'] #<= 24] = 0
    x4 = test_df['num_var37_0'] #> 137615] = 0
    x5 = test_df['saldo_var5'] #> 60099] = 0
    x6 = test_df['saldo_var8']
    x7 = test_df['saldo_var14'] #> 19053.78] = 0
    x8 = test_df['saldo_var17'] #> 288188.97] = 0
    x9 = test_df['saldo_var26'] #> 10381.29] = 0
    x10 = test_df['num_var13_largo_0'] #> 3] = 0
    x11 = test_df['imp_op_var40_comer_ult1'] #> 3639.87] = 0
    x12 = test_df['saldo_medio_var13_largo_ult1'] #> 0] = 0
    x13 = test_df['num_meses_var13_largo_ult3'] #> 0] = 0
    x14 = test_df['num_var20_0'] #> 0] = 0  
    x15 = test_df['saldo_var13_largo'] #> 150000] = 0
    x16 = test_df['num_var17_0'] #> 21
    x17 = test_df['num_var5_0'] #> 6] = 0

    # # preds[var15 < 23] = 0
    # # preds[saldo_medio_var5_hace2 > 160000] = 0
    # # preds[saldo_var33 > 0]=0
    # # preds[var38 > 3988596]=0
    # # preds[V21>7500]=0

    for i in range(len(preds)):
        if var15[i] < 23 or saldo_medio_var5_hace2[i] > 160000 or saldo_var33[i] > 0 or var38[i] > 3988596 or v21[i]>7500 or nv[i] > 0 or num_var30[i] > 9 or num_var13_0[i] > 6 or num_var33_0[i] > 0:
            preds[i] = 0
        if x0[i] > 51003 or x1[i] >13184 or x2[i] > 108251 or x3[i] <= 24 or x4[i] > 45 or x5[i] > 137615 or x6[i] > 60099 or x7[i] > 19053.78 or x8[i] > 288188.97 or x9[i] > 10381.29 :
            preds[i] = 0
        
        if x10[i] > 3 or x11[i] > 3639.87 or x12[i] > 0 or x13[i] > 0 or x14[i] > 0 or x15[i] > 150000 or x16[i] > 21 or x17[i] >6  : 
            preds[i] = 0

        if preds[i] < 0.001:
            preds[i] = 0

    submission = pd.DataFrame({ 'ID': test_df['ID'],'TARGET': preds })
    submission = submission[['ID','TARGET']]
    submission.to_csv("submission_64.csv", index=False)

def rank_solutions():
    x1 = pd.read_csv('submission_logpred_81.csv', header=0)
    x2 = pd.read_csv('submission_new_rank_submit.csv', header=0)
    x3 = pd.read_csv('submission_new_xgb83498.csv', header=0)
    x4 = pd.read_csv('submission_RF.csv', header=0)
    
    a1 = 2 #1 - 0.63
    a2 = 300 #0.63
    a3 = 60
    a4 = 15


    #87699 is original logistic
    #87579 is freq xgb
    #86547 is randomforrests 200, None, entropy
    #88039 is xgb freq_accuracy
    #86410 is ET freq_accuracy
    
    pred1 = np.array(x1['Action'])
    pred2 = np.array(x2['Action'])
    pred3 = np.array(x3['Action'])
    pred4 = np.array(x4['Action'])

    predictions = copy.deepcopy(pred4)

    for i in range(len(predictions)):
        predictions[i] = (5*np.max([pred1[i],pred2[i],pred3[i],pred4[i]]) + 2*np.min([pred1[i],pred2[i],pred3[i],pred4[i]]))/7

    submission = pd.DataFrame({ 'Id': test_df['id'],'Action': predictions })
    submission = submission[['Id','Action']]
    submission.to_csv("submission.csv", index=False)


    

    
    
    
    
    
    
    
    

    
if __name__=="__main__":

    train_df,test_df = load_data('train.csv','test.csv')
    # print train_df.shape
    # print test_df.shape
    # preprocess(train_df,test_df)
    # predictions = holdout(train_df,test_df)
    # write_to_file(test_df,predictions)
    # 
    # new_features(train_df,test_df)
    # predictions = holdout(train_df,test_df)
    # write_to_file(test_df,predictions)

    
    write_test_hack()
    # rank_solutions()
    
    
    
    
	



