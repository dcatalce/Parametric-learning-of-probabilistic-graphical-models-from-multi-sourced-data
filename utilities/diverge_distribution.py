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
from utilities.upload_files import load_compute_edges,remove_constant_columns
from utilities.utils import hier_bayes, define_dag, Laplace,equalFreqBins, compute_theta_MD_mean
from utilities.fit_hierarchical import model,convert2CPD, map_categories,cpds_multi,fit_hierarchical, fit,cpds_all
from utilities.predict import predict



def original_estimates(sample, path, class_s, group, mod, alpha):
    accs_mle = []
    precision_mle = []
    recall_mle = []
    f1_mle = []

    n_samples = []
    acc_datasets = []

    df, edges, features = load_compute_edges(path,class_s,group,mod,nvals= 4, sample = sample)
    dag_orig, dag_hier = define_dag(edges,df, group)
    data = df.copy()
    data = data[features]
    # Define dictionary with possible values of the variables in df to avoid unobserved

    dict_map = map_categories(data)

    Xy_train = data.iloc[:,:]
    x_train= data.loc[:,features]
    data_copy = x_train  
    # New
    dict_values = {}
    for i in data_copy.columns:
        dict_values[i] = list(np.unique(data_copy.loc[:,i]))


    diverged_model = BayesianNetwork(ebunch=dag_orig.edges())    
    diverged_model.fit(data=x_train, estimator=BayesianEstimator,state_names = dict_values, prior_type='BDeu', equivalent_sample_size=alpha)

    x_train_M = x_train.copy()
    x_train_M = x_train_M[x_train_M['gender'] == 'Male']
    len_M = len(x_train_M)
    diverged_model_M = BayesianNetwork(ebunch=dag_orig.edges())    
    diverged_model_M.fit(data=x_train_M, estimator=BayesianEstimator,state_names = dict_values, prior_type='BDeu', equivalent_sample_size=alpha)

    x_train_F = x_train.copy()
    x_train_F = x_train_F[x_train_F['gender'] == 'Female']
    len_F = len(x_train_F)
    diverged_model_F = BayesianNetwork(ebunch=dag_orig.edges())    
    diverged_model_F.fit(data=x_train_F, estimator=BayesianEstimator,state_names = dict_values, prior_type='BDeu', equivalent_sample_size=alpha)

    return diverged_model, diverged_model_F, diverged_model_M, x_train_F, x_train_M, len_M,len_F, edges, features


def diverge_sample(model_M, model_F, model_full, delta, class_s, group):
    model_Male= model_M.copy()
    model_Female= model_F.copy()
    model = model_full.copy()

    for i,var in enumerate(model.nodes): 
        if (var != class_s) & (var != group):    
            dims = model.get_cpds(var).values.shape

            max_var = (model_Male.get_cpds(var).values - model.get_cpds(var).values)
            thetas_M = model_Male.get_cpds(var).values
            thetas_F = model_Female.get_cpds(var).values

            if len(dims)>2:
                for col in range(max_var.shape[1]):
                    for col2 in range(max_var.shape[2]):
                        index_max = np.argmax(np.absolute(max_var[:,col,col2]))
                        s = max_var[index_max,col,col2]
                        theta_0_M = model_Male.get_cpds(var).values[index_max,col,col2]
                        theta_0_F = model_Female.get_cpds(var).values[index_max,col,col2]
                        if s > 0:
                            theta_0_new_M = (1-theta_0_M)*delta +theta_0_M
                            theta_0_new_F = theta_0_F -theta_0_F*delta  
                        else:
                            theta_0_new_F = (1-theta_0_F)*delta +theta_0_F
                            theta_0_new_M = theta_0_M -theta_0_M*delta    

                        numerator_M = 1-theta_0_new_M
                        numerator_F = 1-theta_0_new_F
                        denominator_M = 1-theta_0_M
                        denominator_F = 1-theta_0_F

                        covariate_prop_M = numerator_M/denominator_M
                        covariate_prop_F = numerator_F/denominator_F



                        for j in range(model.get_cpds(var).values.shape[0]):
                            if j != index_max:
                                new_value_gender_M = covariate_prop_M*model_Male.get_cpds(var).values[j,col,col2]
                                model_Male.get_cpds(var).values[j,col,col2] = new_value_gender_M

                                new_value_gender_F = covariate_prop_F*model_Female.get_cpds(var).values[j,col,col2]
                                model_Female.get_cpds(var).values[j,col,col2] = new_value_gender_F                  

                            else:

                                model_Male.get_cpds(var).values[j,col,col2] = theta_0_new_M
                                model_Female.get_cpds(var).values[j,col,col2] = theta_0_new_F
            else:
                for col in range(max_var.shape[1]):
                    index_max = np.argmax(np.absolute(max_var[:,col]))
                    s = max_var[index_max,col]
                    theta_0_M = model_Male.get_cpds(var).values[index_max,col]
                    theta_0_F = model_Female.get_cpds(var).values[index_max,col]
                    if s > 0:
                        theta_0_new_M = (1-theta_0_M)*delta +theta_0_M
                        theta_0_new_F = theta_0_F -theta_0_F*delta  
                    else:
                        theta_0_new_F = (1-theta_0_F)*delta +theta_0_F
                        theta_0_new_M = theta_0_M -theta_0_M*delta    

                    numerator_M = 1-theta_0_new_M
                    numerator_F = 1-theta_0_new_F
                    denominator_M = 1-theta_0_M
                    denominator_F = 1-theta_0_F

                    covariate_prop_M = numerator_M/denominator_M
                    covariate_prop_F = numerator_F/denominator_F


                    for j in range(model.get_cpds(var).values.shape[0]):
                        if j != index_max:
                            new_value_gender_M = covariate_prop_M*model_Male.get_cpds(var).values[j,col]
                            model_Male.get_cpds(var).values[j,col] = new_value_gender_M

                            new_value_gender_F = covariate_prop_F*model_Female.get_cpds(var).values[j,col]
                            model_Female.get_cpds(var).values[j,col] = new_value_gender_F                  

                        else:

                            model_Male.get_cpds(var).values[j,col] = theta_0_new_M
                            model_Female.get_cpds(var).values[j,col] = theta_0_new_F


    return model_Male, model_Female