
    data {
      int<lower=2> yStates;
      int<lower=2> xStates;
      int Ntotal; //number of instances
      vector[yStates] yCounts;
      matrix[xStates,yStates] xyCounts;
      vector[xStates] alpha0; //the highest-level Dir coefficients
      real<lower=0> s; //amount of local smoothing. No prior on s for the moment, to be added later however.
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
    