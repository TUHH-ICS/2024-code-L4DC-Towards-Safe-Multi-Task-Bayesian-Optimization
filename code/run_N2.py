#!/usr/bin/env python3

#---------------------------------------------------------------------------------------------
# For Paper, 
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
#--------------------------------------------------------------------------------------------

# This script can be used to reproduce the data for N = 2 lasers
# Execute script with one addtional argument which contains the path and folder name for the data 

import torch
from botorch.utils.transforms import normalize, unnormalize
import utils.utils
from utils.utils import sample_from_task,concat_data
from utils.mcmc_samples import get_mcmc_samples
import utils.get_robust_gp
from bo.bo_loop import BayesianOptimization

torch.set_default_dtype(torch.float64)

from plant.utils import build_laser_model, get_nh2
from numpy import load

import sys

import pickle

smoke_test = False

torch.set_default_dtype(torch.float64)

if len(sys.argv) > 1:
    add_arg = sys.argv[1]        # Should \bar{beta} be bounded? if so add trunc as additional argument to the script
    if add_arg.find("trunc") != -1: utils.get_robust_gp.change_truncate(True)


# controller typ
K_typ = "PI"

# bounds
Kp_max = 3e1
Kp_min = 2e-1
Ki_max = 3e1
Ki_min = 0

ell = 0.1     # lengthscale
utils.utils.change_ell(ell)


# filter disturbance 0.2 corresponds to +- 10% 
disturbance = 1/5 # for paper plots change to 0/5, 1/5, 3/5, 5/5, 7/5 

# shape factor eta for lkj prior 
if disturbance == 0 or disturbance == 1/5: 
    eta = 0.1 
elif disturbance == 3/5: 
    eta = 0.5
else: 
    eta = 1.0
utils.utils.change_eta(eta)
  

num = 2         # number of lasers in chain "N"
num_tsk = 3     # number of tasks "u"
tasks = list(range(num_tsk))

bounds=torch.tensor([[Kp_min,Ki_min],[Kp_max,Ki_max]]).repeat(1,num)

X_init = load("data_paper/X_init_vals_num2.npy")
x0 = torch.tensor(X_init[0,...]).unsqueeze(0)

if K_typ == "PI":
    bounds=torch.tensor([[Kp_min,Ki_min],[Kp_max,Ki_max]]).repeat(1,num)
else:
    bounds=torch.tensor([[Kp_min],[Kp_max]]).repeat(1,num)
T = -30

nruns = 2        # number of max main task evaluations (can be reduced)
d = bounds.size(-1) # input dimension
init = 7

data_sets=[]
bests = []
Gges = []
X_init = torch.tensor(X_init)
for i in range(X_init.size(0)):
    x0 = X_init[init,...].view(1,bounds.size(-1))

    G = [build_laser_model(num, disturbance = 0. if i == 0 else disturbance) for i in range(num_tsk)]
    Gges.append(G)
    obj = [lambda param, G=G[i]: get_nh2(param,G,bounds,K_typ) for i in range(num_tsk)]

    norm_bounds = torch.vstack((torch.zeros(1,d),torch.ones(1,d)))
    norm_x0 = normalize(x0,bounds)

   # evalaute initial point for all tasks
    train_targets = torch.zeros(num_tsk,1)
    for i in range(num_tsk):
        train_targets[i,...] = obj[i](norm_x0)
    train_tasks = torch.arange(num_tsk).unsqueeze(-1)
    train_inputs = norm_x0.repeat(num_tsk,1)

    # evalaute supplementary tasks
    for i in range(1,num_tsk):
        x, t, y = sample_from_task(obj,[i],norm_bounds,n=20)
        train_inputs, train_tasks, train_targets = concat_data((x, t, y),(train_inputs,train_tasks,train_targets))

    gp = utils.utils.build_gp((train_inputs,train_tasks),train_targets)
    bo = BayesianOptimization(obj,tasks,norm_bounds,T,[1,15,15])

    for i in range(nruns):
        if i%2 == 0:
            sample_models = get_mcmc_samples(gp=gp)
            robust_gp,sqrtbeta,gamma = utils.get_robust_gp.getboundinggp(sample_models,gp)
            del sample_models

            bo.update_gp(robust_gp,sqrtbeta)
        train_inputs,train_tasks,train_targets = bo.step()
        gp = utils.utils.build_gp((train_inputs,train_tasks),train_targets)
    data_sets.append([train_inputs,train_tasks,train_targets])
    bests.append([bo.best_x,bo.best_y])
    print(f"Best value: {round(bo.best_y[-1],3)} at input: {unnormalize(bo.best_x[-1],bounds).round(decimals=3)}")

    file = open(f"data/num_laser{num}_dist{int(5*disturbance)}.obj",'wb')
    sets = {'data_sets': data_sets, 'bests': bests, 'plants': Gges}
    pickle.dump(sets,file)
    file.close()


