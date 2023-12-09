%---------------------------------------------------------------------------------------------
% For Paper, 
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Author(s): Jannis Lübsen, Maximilian Schütte
%--------------------------------------------------------------------------------------------
% Define boundaries
cond_r=[...];
cond = repmat([-1,1],size(cond_r,1),1);

inf_ = {@infGaussLik};
mean_ = {@meanConst};
lik_ = {@likGauss};
cov_ = {@covMaternard,3};

% Define initial hyperparameter
hyp.lik = log(0.4);
hyp.mean = 50;
hyp.cov = log([(cond(:,2)-cond(:,1)).*scale;15]);

% Define acquisition functiom
acq = {@EI};


x0 = forwardCoordTransf(cond_r,...); %plug in initail safe point

opts.plot=1; % enable plot
opts.minFunc.mode=3; 
opts.acqFunc.xi = 0.0;
opts.acqFunc.beta = 2;

% Condition when the GP should be fitted (also minizers are considered
opts.trainGP.acqVal = 10;
% Define problem: maximization = 1, minimization = 0
opts.maxProb = 0;
% Exploitation termination condition
opts.termCondAcq = 0.1;
% Hard termination condition for one subspace
opts.maxIt = 500;
% Iterations when GP should be fitted (only used when safe options are
% disabled
opts.trainGP.It = 501;
opts.trainGP.train = 0;
% Enable both to use moSaOpt
opts.safeOpt = 1;
opts.moSaOpt = 1;
% Define threshold and its decay to the boundaries
opts.safeOpts.threshold = 50;
opts.safeOpts.thresholdOffset = 8;
opts.safeOpts.thresholdPer = 0.2;
opts.safeOpts.thresholdOrder = 1;
% Termination condition for the v reachable set
opts.safeOpts.searchCond = 3;
% Alternatively exploration is terminated after explorationIt iterations
opts.safeOpts.explorationIt = 50;
% Expand only in diretions where expanders are also minimizers
opts.safeOpts.onlyOptiDir = false;
% Maximum number of subspace iterations
opts_lBO.maxIt = 100;
% Use a shared GP between subspaces
opts_lBO.sharedGP = 1;

% subspaceDim = 2 to use PlaneBO
opts_lBO.subspaceDim = 1;

% - Condition whether a space
% - could be optimized
% opts_lBO.obj_eval = @(y1,y2) y1 > y2+0.1;  
% - directory to save time data
%opts.dir_timeData="";
% - restrict subspace combinations eg. [1;2;5;6] or [1,2;3,4;5,6]
% opts_lBO.dim_combinations = [2;6];
% - uncomment to use descent oracle
% opts_lBO.oracle = 'descent';

time = 0;
% Define a function with one output with a function handle
fun = ...
tic
% Execute lineBO or bayesOptima (if you want to optimize over the whole
% space once
[xopt,X,Y,DIM]=lineBO(hyp,inf_,mean_,cov_,lik_,acq, fun,cond,opts,opts_lBO,x0);
%[xopt,yopt,X,Y]=bayesOptima(hyp,inf_,mean_,cov_,lik_,acq,...
%fun,cond,opts,x0 (otherwise randomly from cond));
toc


        
        
    
