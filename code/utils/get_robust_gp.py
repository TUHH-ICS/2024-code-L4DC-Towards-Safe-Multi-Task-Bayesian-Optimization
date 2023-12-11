#!/usr/bin/env python3

#---------------------------------------------------------------------------------------------
# For Paper, 
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
#--------------------------------------------------------------------------------------------

import torch
from utils.task_gamma import get_task_gamma
from utils.get_mean_norms import get_mean_norms
from copy import deepcopy

truncate = False


def change_truncate(x):
    global truncate
    truncate = x


def getboundinggp(sampmods, model0, delta_max: float = 0.05):
    gamma, task_lambda, task_thdoubprime = get_task_gamma(model0, sampmods, delta_max)

    print(f"Robust Task Covar Factor: {task_thdoubprime}")
    robustmodel = deepcopy(model0)

    robustmodel.task_covar_module._set_covar_factor(task_thdoubprime)
    # sqrtzeta = get_mean_norms(model0,robustmodel,task_lambda) TODO: not fully supported yet

    maxsqrtbeta = 2

    sqrtbeta = gamma * maxsqrtbeta  # + sqrtzeta
    if truncate:
        sqrtbeta = torch.minimum(sqrtbeta, torch.tensor([2 * maxsqrtbeta]))
    # print(f"sqrtbeta: {sqrtbeta}")
    # print(f"sqrtzeta: {sqrtzeta}")
    # print(f"Covar_Mat: {task_thdoubprime@task_thdoubprime.T}")
    # print(f"gamma: {gamma}")

    return robustmodel, sqrtbeta.squeeze(), gamma.squeeze()
