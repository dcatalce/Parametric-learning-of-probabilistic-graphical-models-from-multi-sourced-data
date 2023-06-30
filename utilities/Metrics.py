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




def plot_roc_curve(y_test, y_probs, method):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("./images/" + file +"/ROC_"+str(mod)+"_"+method+"_alpha_"+str(alpha)+file+".png" )
    plt.show()
    plt.close()


def calculate_metrics(y_test,y_probs, y_pred, file, mod, alpha ,lev = None):

    
    acc = np.round(metrics.accuracy_score(y_test,y_pred),4)
    print("Accuracy of the model:",acc)
    
    precision = np.round(metrics.precision_score(y_test,y_pred),3)
    recall = np.round(metrics.recall_score(y_test,y_pred),3)
    f1_score = np.round(metrics.f1_score(y_test,y_pred),3)
    print(
    "Precision:",precision,"\n",
    "Recall:",recall,"\n",
    "F1 Score:",f1_score,"\n" )
    
    return acc, precision, recall, f1_score

