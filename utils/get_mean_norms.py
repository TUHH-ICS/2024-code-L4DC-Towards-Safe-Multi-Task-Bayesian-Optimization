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

def get_mean_norms(model0,model,lambda_):
    model0.train()
    model.train()
    train_inputs = model0.train_inputs[0]
    y = model0.train_targets.unsqueeze(-1)
    S0 = model0(train_inputs)
    K0 = S0.covariance_matrix
    IK01 = torch.linalg.solve(model0.likelihood(S0).covariance_matrix,torch.eye(K0.size(0)))
    S1 = model(train_inputs)
    K1 = S1.covariance_matrix
    IK11 = torch.linalg.solve(model.likelihood(S1).covariance_matrix,torch.eye(K1.size(0)))
    return torch.sqrt(y.T@(lambda_**2*IK01@K0@IK01-2*IK01@K1@IK11+IK11@K1@IK11)@y)

#TODO Add the term of the posterior means!