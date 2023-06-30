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
import logging


def VB_stan_hierMD(xValues, yValues, xStates, yStates, levelsy, xyCounts = False):
    
    
    training_samples = len(xValues)
    yCounts = pd.Series(yValues).value_counts()
    xCounts = pd.Series(xValues).value_counts()

    for z in levelsy:
        if z not in yCounts.index:
            yCounts[z] = 0 

    datadict = dict(
    yStates = yStates,
    xStates = xStates,
    Ntotal = training_samples,
    yCounts = yCounts,
    xCounts = xCounts,
    xyCounts = xyCounts,
    alpha0 = np.repeat(1,xStates),
    s =  xStates)
    
    simple_model_code = """"

    data {
      int<lower=2> yStates;
      int<lower=2> xStates;
      int Ntotal; //number of instances
      vector[yStates] yCounts;
      matrix[xStates,yStates] xyCounts;
      vector[xStates] alpha0; //the highest-level Dir coefficients
      real<lower=0> s; //amount of local smoothing.
    }


    parameters {
    simplex[yStates] thetaY;
    simplex[xStates] thetaX[yStates];
    simplex[xStates] alpha;//the prior  for the local states
    }

    model {

    //sample the thetaY from the posterior  Dirichlet
    thetaY ~ dirichlet (1.0 + yCounts);

    //we treat alpha0 as a fixed vector of ones
    alpha   ~ dirichlet (alpha0);

    for (y in 1:yStates){
    //sample the thetaXY from a Dir(1,1,1,...1)
    thetaX[y] ~ dirichlet (s*alpha + col(xyCounts,y));
    }
    }
    """

    logger = logging.getLogger('cmdstanpy')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    model = CmdStanModel(stan_file="./models/simple_model.stan")
    vb = model.variational(data=datadict,require_converged=False)
    
    
    vb_sample = vb.variational_sample.T
    vb_parameters = vb.variational_params_dict
    
    post_samples = len(vb_sample.T)
    columns = list(vb.column_names)
    currentThetaY = np.zeros(yStates)
    currentThetaX = np.zeros(xStates)
    currentThetaXgivenY = np.zeros([xStates,yStates])
    index_thetaY = [columns.index(i) for i in columns if 'thetaY' in i]
    index_thetaX = [columns.index(i) for i in columns if 'thetaX' in i]
    index_alpha = [columns.index(i) for i in columns if 'alpha' in i]
    samples_thetaY = vb_sample[index_thetaY]
    samples_thetaXY = vb_sample[index_thetaX]
    samples_alpha = vb_sample[index_alpha]
    
    theta = samples_thetaXY
    thetaY = samples_thetaY
    alpha = samples_alpha
    
    return dict(theta = theta, thetaY = thetaY, alpha = alpha)