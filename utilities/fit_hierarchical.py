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
from utilities.utils import hier_bayes, define_dag, Laplace,equalFreqBins, compute_theta_MD_mean

def model(dag_hier,dag_orig,data, alpha, class_s, train, method):
    group = 'gender'
    if method == 'multi-data':
        models = {}
        for n,lev in enumerate(['Female','Male']):
            dag_orig_multi= dag_orig.copy()
            dag_hier_multi = dag_hier.copy()
            thetas = {}
            for node in list(dag_orig.nodes):
                theta, levels = fit_hierarchical(node, alpha, group,data, train,dag_hier_multi,dag_orig_multi)
                thetas[node] = theta
            dag_orig = cpds_multi(dag_orig_multi,thetas, data, class_s, lev)
            models[lev] = dag_orig_multi
    elif method == 'new':
        thetas = {}

        for node in list(dag_hier.nodes):
            theta = fit(node, alpha,data, train,dag_hier)
            thetas[node] = theta
        #return thetas
        models = cpds_all(dag_hier,thetas, data)
    
    else:       
        thetas = {}
        for node in list(dag_orig.nodes):
            theta = fit(node, alpha,data, train,dag_orig)
            thetas[node] = theta
        models = cpds_all(dag_orig,thetas, data)

    return models
      

def convert2CPD(theta_node,levelsy,node,dag):
    
    new_thetas = {}
    for j in levelsy:
        if len(dag.get_parents(node)) == 3:
            theta = theta_node.copy()
            first_var = [i.split('/')[0] for i in theta.index]
            second_var = [i.split('/')[1] for i in theta.index]
            third_var = [i.split('/')[2] for i in theta.index]
            theta[node] = first_var
            theta[dag.get_parents(node)[1]] = second_var
            theta[dag.get_parents(node)[0]] = third_var
            remove = [i for i in levelsy if i != j]
            theta.reset_index(drop = True, inplace = True)
            theta.drop(remove, axis = 1, inplace = True)
            theta = theta.pivot(index =node, columns =[dag.get_parents(node)[1],dag.get_parents(node)[0]])
            marg = np.sum(theta, axis = 0)
        else:
            theta = theta_node.copy()
            first_var = [i.split('/')[0] for i in theta.index]
            second_var = [i.split('/')[1] for i in theta.index]
            theta[node] = first_var
            theta[dag.get_parents(node)[1]] = second_var
            remove = [i for i in levelsy if i != j]
            theta.reset_index(drop = True, inplace = True)
            theta.drop(remove, axis = 1, inplace = True)
            theta = theta.pivot(index =node, columns =[dag.get_parents(node)[1]])
            marg = np.sum(theta, axis = 0)
            

    
        for i in range(len(theta.columns)):
            theta.iloc[:,i] = theta.iloc[:,i]/marg.iloc[i]
         
        new_thetas[j] = theta
        
    return new_thetas

def map_categories(data):
    factorized_values = [np.unique(pd.factorize(data[x], sort = True)[1]) for x in data.columns]
    factorized_index = [np.unique(pd.factorize(data[x], sort = True)[0]) for x in data.columns]
    list_dicts = []
    for i in range(len(factorized_values)):
        dict_a = {}
        for x,j in zip(factorized_values[i],factorized_index[i]):
            dict_a[x] = j
        list_dicts.append(dict_a)
        
    return list_dicts

def cpds_multi(dag_orig,thetas, full_data, class_s, lev):

    group = 'gender'   
    for node in thetas.keys():
        if node == group:
            pass

        elif (type(thetas[node]) == dict) & (len(dag_orig.get_parents(node)) == 1):
            cpd = TabularCPD(
                        variable = node,
                        variable_card = len(thetas[node][lev]),
                        values = [thetas[node][lev].iloc[i,:].values for i in range(thetas[node][lev].shape[0])],
                        evidence = dag_orig.get_parents(node), evidence_card = [thetas[node][lev].shape[1]])
            dag_orig.add_cpds(cpd)

        elif (type(thetas[node]) == dict) & (len(dag_orig.get_parents(node)) == 2):
            cpd = TabularCPD(
                        variable = node,
                        variable_card = len(thetas[node][lev]),
                        values = [thetas[node][lev].iloc[i,:].values for i in range(thetas[node][lev].shape[0])],
                        evidence = [dag_orig.get_parents(node)[1], dag_orig.get_parents(node)[0]],
                        evidence_card = [len(pd.unique(full_data[dag_orig.get_parents(node)[1]])),
                                        len(pd.unique(full_data[dag_orig.get_parents(node)[0]]))])
            dag_orig.add_cpds(cpd)


        elif node == class_s:
            cpd = TabularCPD(
                        variable = node, 
                        variable_card = thetas[node].shape[0],
                        values = [[thetas[node][lev].iloc[i]] for i in range(thetas[node].shape[0])])

            dag_orig.add_cpds(cpd)
        else:
            pass


    return dag_orig


def fit_hierarchical(node, alpha, group, full_data, data,dag_hier,dag_orig):
    laplace = True
    # Parents
    parents = dag_hier.get_parents(node = node)
    #Childs
    childs = dag_hier.get_children(node = node)
    
    if len(parents) > 0:
        yValues = data[parents[0]]  
        newstates = data[node] 
        
        levelsx = full_data[node].astype('category').cat.categories
        levelsy = full_data.loc[:,parents[0]].astype('category').cat.categories            
        
        if len(parents) > 1:
            j = 1

            levels_new = np.unique(data[parents[j-1]]) # unique valores  del padre no grupo 
            if len(parents)> 2:
                j = 2
                other_level = full_data.loc[:,parents[j-1]].astype('category').cat.categories
                levelsx = np.array([(x+ "/" + y + "/"+ z) for x in levelsx for y in other_level for z in levelsy] ) # valores de X y del padre no grupo
                newstates = newstates + "/" +  data[parents[j-1]] + "/" + data[parents[j-2] ]    
                
            else:
                levelsx = np.array([(x+ "/" + y ) for x in levelsx for y in levelsy ] ) # valores de X y del padre no grupo
                newstates = newstates + "/" +  yValues    

        
        xStates = len(levelsx)
        yStates = len(np.unique(data[group])) # Valores de la variable auxiliar (nuevo padre)
        levelsy = np.unique(data[group])
        yValues = data[group] 
        
        xyCounts = pd.DataFrame(np.zeros([xStates,yStates]), columns = levelsy, index = levelsx)
        for y in levelsy:
            for j in levelsx:
                xyCounts.loc[j][y] =  len(newstates[(newstates == j) &(yValues == y)]) +0.01

               
        dirichlet_prior = np.repeat(1,xStates)
        
        xValues = newstates        
        yValues = data[group]
        
        if laplace:
            s = alpha*xStates
        else:
            s = alpha/yStates     
        
        VB_stan_model = VB_stan_hierMD(xValues, yValues, xStates , yStates, levelsy, xyCounts )
        theta_VB = np.mean(VB_stan_model['theta'], axis = 1)
        theta = theta_VB

        # Se genera una matriz de la combinación de X con sus Pa, donde la variable auxiliar
        # son las Columnas (es padre de todos). Por tanto es una Joint distribution de X
        # y sus padres Pa (incluyendo la clase) condicionado por la variable auxiliar F. 
        MAT = np.zeros([xStates,yStates])

        for x in range(1, xStates):
            MAT[0,0:yStates] = theta[0:yStates]
            MAT[x,0:yStates] = theta[(yStates*x):yStates*(x+1)]

        MAT = pd.DataFrame(MAT, columns = levelsy, index = levelsx)
        if len(parents) > 1:
            # Convert2CPD lo que hace es calcular tantas tablas de distribuciones de probabilidad
            # Como valores de la variable auxiliar f haya.
            MAT = convert2CPD(MAT,levelsy,node, dag_hier)
        
        
    else:
        levelsy = np.unique(data[group])
        xValues = data[node]
        levelsx = np.unique(data[node])
        xStates = len(levelsx)
        yStates = 1
        xCounts = pd.Series(xValues).value_counts()

        if laplace:
            s = alpha*xStates
        else:
            s = alpha/yStates

        theta = compute_theta_MD_mean(xCounts,s)
        theta = pd.DataFrame([theta], columns = levelsx)
        MAT = theta
        meaning_xStates = levelsx
                
    return MAT, levelsy




def fit(node, alpha,full_data, data, dag):
    laplace = True
    # Parents
    parents = dag.get_parents(node = node)
    #Childs
    childs = dag.get_children(node = node)

    
    if len(parents) > 0:
        
        xValues = data[node] 
        yValues = data[parents[0]] 

        levelsx = full_data[node].astype('category').cat.categories
        
        levelsy = full_data.loc[:,parents[len(parents)-1]].astype('category').cat.categories # levels relationship
                                                            
        xStates = len(levelsx)
        yStates = len(levelsy)
        newstates = data.loc[:,parents[len(parents)-1]] # relationship

        xyCounts = (pd.crosstab(newstates, xValues, dropna = False) + 0.01).T
        if len(parents) > 1:
            if len(parents) == 2:
                other_level = data.loc[:,parents[len(parents)-2]]
                yValues = newstates + "/"+ other_level
                levelsy = np.array([(x+ "/" + y) for x in levelsy for y in np.unique(full_data[parents[len(parents)-2]])] )
                xStates = len(levelsx)
                yStates = len(levelsy)
                # levels relationship
            elif len(parents) == 3:
                other_level_1 = data.loc[:,parents[len(parents)-2]]
                other_level_2 = data.loc[:,parents[len(parents)-3]]
                levelsy = np.array([(x+ "/" + y + "/" + z) for x in levelsy for y in np.unique(full_data[parents[len(parents)-2]]) for z in np.unique(full_data[parents[len(parents)-3]])] ) # relationship + income 
                yValues = newstates + "/" +  other_level_1 + "/" + other_level_2    # levels of relationship + income 
                xStates = len(levelsx)
                yStates = len(levelsy)
        xyCounts = pd.DataFrame(np.zeros([xStates,yStates]), columns = levelsy, index = levelsx)
        
        for y in levelsy:
            for x in levelsx:
                xyCounts.loc[x][y] =  len(yValues[(yValues == y) &(xValues == x)]) +0.01   

        dirichlet_prior = np.repeat(1,xStates)

        if laplace:
            s = alpha*xStates
        else:
            s = alpha/yStates
            
        VB_stan_model = VB_stan_hierMD(xValues, yValues, xStates , yStates, levelsy, xyCounts )
        
        theta_VB = np.mean(VB_stan_model['theta'], axis = 1)
        theta = theta_VB
        MAT = np.zeros([xStates,yStates])
        
        for x in range(1, xStates):
            MAT[0,0:yStates] = theta[0:yStates]
            MAT[x,0:yStates] = theta[(yStates*x):yStates*(x+1)]
            
        theta = pd.DataFrame(MAT, columns = levelsy, index = levelsx)

        
    else:
        xValues = data[node]
        levelsx = np.unique(data[node])
        xStates = len(levelsx)
        yStates = 1
        xCounts = pd.Series(xValues).value_counts()

        if laplace:
            s = alpha*xStates
        else:
            s = alpha/yStates

        theta=compute_theta_MD_mean(xCounts,s)
        theta = pd.DataFrame([theta], columns = levelsx)
    
    return theta


def cpds_all(dag, thetas, data):
    for node in thetas.keys():
        if len(dag.get_parents(node)) ==1:

            cpd = TabularCPD(
                        variable = node,
                        variable_card = thetas[node].shape[0],
                        values = [thetas[node].iloc[i,:].values for i in range(thetas[node].shape[0])],
                        evidence = dag.get_parents(node), evidence_card = [thetas[node].shape[1]])
            dag.add_cpds(cpd)
            
        elif len(dag.get_parents(node)) ==2:

            cpd = TabularCPD(
                                variable = node, 
                                variable_card = thetas[node].shape[0],
                                values = [thetas[node].iloc[i,:].values for i in range(thetas[node].shape[0])],
                                evidence = [dag.get_parents(node)[1], dag.get_parents(node)[0]],
                                evidence_card = [len(pd.unique(data[dag.get_parents(node)[1]]))
                                                 ,len(pd.unique(data[dag.get_parents(node)[0]]))])

            dag.add_cpds(cpd)
        
        elif len(dag.get_parents(node)) ==3:
            cpd = TabularCPD(
                                variable = node, 
                                variable_card = thetas[node].shape[0],
                                values = [thetas[node].iloc[i,:].values for i in range(thetas[node].shape[0])],
                                evidence = [dag.get_parents(node)[2],dag.get_parents(node)[1], dag.get_parents(node)[0]],
                                evidence_card = [len(pd.unique(data[dag.get_parents(node)[2]])),
                                                 len(pd.unique(data[dag.get_parents(node)[1]])),
                                                 len(pd.unique(data[dag.get_parents(node)[0]]))])

            dag.add_cpds(cpd)
            
        else: 
            cpd = TabularCPD(
                            variable = node, 
                            variable_card = thetas[node].shape[1],
                            values = [[thetas[node].iloc[0,i]] for i in range(thetas[node].shape[1])])
            dag.add_cpds(cpd)

    return dag