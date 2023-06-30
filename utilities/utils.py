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

def Laplace(model, pseudocounts = 0.01):
    fitted_model = model.copy()
    # Assume that you have already fitted the model and obtained the fitted CPDs
    fitted_cpds = fitted_model.get_cpds()

    # Apply Laplace smoothing to the fitted CPDs
    pseudocount = 1

    for cpd in fitted_cpds:
        variable = cpd.variable
        state_names = cpd.state_names[variable]
        cpd.values += pseudocount
        cpd.normalize()

    # Update the CPDs in the fitted model
    fitted_model.add_cpds(*fitted_cpds)
    return fitted_model

def hier_bayes(dag, group,new_dag, nodes):
    # Creates a new DAG with group as parent variable for every node (only for method="multi-domain")

    # dag = dag structure in bnlearn format
    # group = name of the variable associated to the domain (only for method="multi-domain")

    for node in nodes:
        if node != group:
            
            new_dag.add_edge(group, node)

    return new_dag


def define_dag(edges,df, group):
    dag = BayesianNetwork(edges)
    new_dag = BayesianNetwork(edges)
    nodes = dag.nodes
    levelsy = np.unique(df[group])

    dag_hier = hier_bayes(dag,group,new_dag, nodes)
    dag_orig = dag
    dag = dag_hier
    nodes = dag.nodes
    edges = dag.edges
    
    return dag_orig, dag_hier


def compute_theta_MD_mean(N,eta0):


    Ydim=len(N)
    theta_mean=(N+eta0/Ydim)/(sum(N)+eta0)
    
    return(theta_mean)

def equalFreqBins(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1)[1:-1],
                     np.arange(nlen),
                     np.sort(x))
