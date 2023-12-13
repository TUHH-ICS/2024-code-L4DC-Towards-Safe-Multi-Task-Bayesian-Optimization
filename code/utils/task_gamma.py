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
from gpytorch.mlls import ExactMarginalLogLikelihood


weight = 10  # weight for selecting parameter bounds


def get_task_gamma(model0, sampmods, delta_max: float):
    covar0 = model0.task_covar_module._eval_covar_matrix().maximum(torch.tensor([0.0]))
    covar = sampmods.task_covar_module._eval_covar_matrix()
    covar = covar.maximum(torch.tensor([0.0]))
    num_tasks = covar0.size(0)

    covar_full = torch.cat((covar, covar0.unsqueeze(0)))
    indmax = round(covar.size(0) * (1 - delta_max))

    # Estimate MAP covar_factor hyperparameter
    # mll = ExactMarginalLogLikelihood(sampmods.likelihood, sampmods)
    # sampmods.train()
    # output = sampmods(sampmods.train_inputs[0])
    # loss = mll(output,sampmods.train_targets)
    # Id = loss.argmax()

    dets = torch.linalg.det(covar_full)
    samps = covar_full.shape[0]

    covar0_t = torch.empty_like(covar0).copy_(covar0)
    covar_full_t = torch.empty_like(covar_full).copy_(covar_full)

    gamma_t = torch.zeros(samps)
    lambda_t = torch.zeros(samps)
    for i in range(samps):
        covar_m = covar_full_t[i, ...]
        flag1 = False
        flag2 = False
        while not flag1:
            tmp, flag1 = _solve_systems(covar_m, covar_full_t)
            gamma_t[i] = (
                torch.linalg.eigvals(tmp).real.max(dim=-1)[0].sort()[0][indmax].sqrt()
            )
            covar_m += torch.eye(num_tasks) * 1e-8
            covar_full_t += torch.eye(num_tasks) * 1e-8
        while not flag2:
            tmp, flag2 = _solve_systems(covar0_t, covar_m)
            lambda_t[i] = torch.linalg.eigvals(tmp).real.max(dim=-1)[0].sqrt()
            covar_m += torch.eye(num_tasks) * 1e-8
            covar0_t += torch.eye(num_tasks) * 1e-8

    _, inds = (gamma_t + weight * dets).sort()
    sorted_gamma = gamma_t[inds]
    flag = False
    c = 0
    while not flag:
        Id = inds[0]
        gamma = sorted_gamma[0]
        lambda_ = lambda_t[0]
        thdoubprime, flag = _get_chol_fact(covar_full[Id])
        covar_full[Id] += torch.eye(num_tasks) * 1e-8
        c += 1
        if c == samps:
            ValueError("Could not find cholesky decomposition")
    return gamma, lambda_, thdoubprime


def _get_chol_fact(mat):
    try:
        thdoubprime = torch.linalg.cholesky(mat)
    except:
        Warning("Chol decomposition failed. Trying next matrix...")
        return torch.tensor([0.0]), False
    return thdoubprime, True


def _solve_systems(A, B):
    try:
        sol = torch.linalg.solve(A, B)
    except:
        return torch.tensor([0.0]), False
    return sol, True
