import pandas as pd
import numpy as np
import os
from hashlib import md5
import cmdstanpy as ps
from scipy.stats import norm,dirichlet, multinomial, beta, binom
import copy
import matplotlib.pyplot as plt
import bnlearn as bn
import pgmpy as pgm
from pgmpy.models import BayesianNetwork
from cmdstanpy import CmdStanModel
from VB import VB_stan_hierMD
from pgmpy.factors.discrete import TabularCPD 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
import networkx as nx
import pylab as plt
import networkx
import matplotlib
import matplotlib.pyplot
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from utilities.utils import hier_bayes, define_dag, Laplace,equalFreqBins

def load_compute_edges(path,class_s,group,structure,nvals= 3,sample = 1):
    df = pd.read_csv(path).replace('?', np.nan)
    if path == "./Datasets/dataset_diabetes/diabetes.csv":
        df = df[df['readmitted'] != 'NO'].reset_index(drop = True)
        columns = ['race','gender','age','time_in_hospital','num_procedures','num_medications','number_outpatient',
                  'number_emergency','number_inpatient','A1Cresult','metformin','chlorpropamide','glipizide',
                  'rosiglitazone','acarbose','miglitol','diabetesMed','readmitted']
        df = df[columns]
        

    # Fill NaN with Mode
    for column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    for column in df.columns:
        try: 
            df[column] = df[column].astype(float)
        except:
            pass

    new_cols = [col for col in df.columns if df[col].dtype != 'object']
    nvals=4
    for j in new_cols:
        df.loc[:,j] = np.digitize(df.loc[:,j],bins=equalFreqBins(df.loc[:,j],nvals))
        df.loc[:,j] = df.loc[:,j].astype('str')
    # Remove constant columns
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]    
    features = list(df.columns)
    cols = [col for col in df.columns if (col != group) & (col != class_s)]
    edges = []
    if structure == 'tan':
        for col in range(len(cols)-1):
            a = tuple([cols[col],cols[col+1]])
            edges.append(a)
        for col in range(len(cols)):
            b = tuple([class_s, cols[col]])
            edges.append(b)
    elif structure == 'naive':
        for col in range(len(cols)):
            b = tuple([class_s, cols[col]])
            edges.append(b)
        
    return df, edges, features


def remove_constant_columns(df,structure, group, class_s):
# Remove constant columns
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]    
    features = list(df.columns)
    cols = [col for col in df.columns if (col != group) & (col != class_s)]
    edges = []
    if structure == 'tan':
        for col in range(len(cols)-1):
            a = tuple([cols[col],cols[col+1]])
            edges.append(a)
        for col in range(len(cols)):
            b = tuple([class_s, cols[col]])
            edges.append(b)
    elif structure == 'naive':
        for col in range(len(cols)):
            b = tuple([class_s, cols[col]])
            edges.append(b)
        
    return df, edges, features