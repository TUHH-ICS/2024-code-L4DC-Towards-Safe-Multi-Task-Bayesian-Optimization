#!/usr/bin/env python3
#---------------------------------------------------------------------------------------------
# For Paper, 
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
#--------------------------------------------------------------------------------------------

# Use this script for online visualization of the optimiztation process for low dimensional problems

import torch
from botorch.utils.transforms import unnormalize
import utils.utils
from utils.utils import sample_from_task, concat_data, plot_post, plot_post2D
from utils.mcmc_samples import get_mcmc_samples
import utils.get_robust_gp
from bo.bo_loop import BayesianOptimization

torch.set_default_dtype(torch.float64)

from plant.utils import build_laser_model, get_nh2

# controller typ
K_typ = "PI" # can be switched to "P"

# bounds
Kp_max = 3e1
Kp_min = 2e-1
Ki_max = 3e1
Ki_min = 0

ell = 0.1         # lengthscale
disturbance = 1/5 # filter disturbance 0.2 corresponds to +- 10%
utils.utils.change_ell(ell)

num = 1         # number of lasers in chain "N"
num_tsk = 3     # number of tasks "u"
tasks = list(range(num_tsk))

if K_typ == "PI":
    bounds=torch.tensor([[Kp_min,Ki_min],[Kp_max,Ki_max]]).repeat(1,num)
else:
    bounds=torch.tensor([[Kp_min],[Kp_max]]).repeat(1,num)

d = bounds.size(-1)
norm_bounds = torch.vstack((torch.zeros(1,d),torch.ones(1,d)))

G = [build_laser_model(num, disturbance = 0. if i == 0 else disturbance) for i in range(num_tsk)]
obj = [lambda param, G=G[i]: get_nh2(param,G,bounds,K_typ) for i in range(num_tsk)]

T = -30


nruns = 10
bo = BayesianOptimization(obj,tasks,norm_bounds,T,[1,15,15],boundary_T=-15)               # Define BO object 1 sample from main, 15 from supplementary tasks

pot_start, _, pot_start_targets = sample_from_task(obj,[0],norm_bounds, n=20)
ind = pot_start_targets[pot_start_targets.squeeze()>=bo._det_threshold(pot_start)].squeeze().argmin()

norm_x0 = pot_start[pot_start_targets.squeeze()>=bo._det_threshold(pot_start),...][ind,...].view(1,d)

train_targets = torch.zeros(num_tsk,1)
for i in range(num_tsk):
    train_targets[i,...] = obj[i](norm_x0)
train_tasks = torch.arange(num_tsk).unsqueeze(-1)
train_inputs = norm_x0.repeat(num_tsk,1)
for i in range(1,num_tsk):
    x, t, y = sample_from_task(obj,[i],norm_bounds,n=20)
    train_inputs, train_tasks, train_targets = concat_data((x, t, y),(train_inputs,train_tasks,train_targets))

gp = utils.utils.build_gp((train_inputs,train_tasks),train_targets)     # build GP model

for i in range(nruns):
    #gp = multistart_optimization(gp,1,mode=1) # fit GP; not done in paper
    if i > 50:
        bo.num_acq_samps=[1,1,1]
    if i%2 == 0:
        sample_models = get_mcmc_samples(gp=gp)
        robust_gp,sqrtbeta,gamma = utils.get_robust_gp.getboundinggp(sample_models,gp)
        if d == 1:
            test_inputs = torch.linspace(0,1,101)
            plot_post(robust_gp,[0],test_inputs,sqrtbeta)
        if d == 2:
            plot_post2D(robust_gp,[0],norm_bounds,sqrtbeta=sqrtbeta)
        del sample_models

        bo.update_gp(robust_gp,sqrtbeta)
    train_inputs,train_tasks,train_targets = bo.step()
    gp = utils.utils.build_gp((train_inputs,train_tasks),train_targets)

print(f"Best value: {round(bo.best_y[-1],3)} at input: {unnormalize(bo.best_x[-1],bounds).round(decimals=3)}")


