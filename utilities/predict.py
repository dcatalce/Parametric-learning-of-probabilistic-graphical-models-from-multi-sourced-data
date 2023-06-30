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


def predict(df, model, dict_map,class_s, group ,lev = None, method = None):
    
    predict_data = df.copy()

    if method == 'multi-data':

        predict_data = predict_data[predict_data[group] == lev].reset_index(drop = True)

        # Map the test 
    print("before map")
    for i,j in enumerate(predict_data.columns):
        predict_data[j] = predict_data[j].map(dict_map[i])
    print("after map")
    y_true = predict_data[class_s]
    print("start predicting")
    if method != "new":
        predict_data.drop([class_s, group], axis=1, inplace=True, errors = 'ignore')
    else:
        predict_data.drop([class_s], axis=1, inplace=True, errors = 'ignore')

    y_probs = model.predict(predict_data)
    print("predicted")
    y_pred = y_probs.values
    return y_true, y_pred, y_probs