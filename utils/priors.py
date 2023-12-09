#---------------------------------------------------------------------------------------------
# For Paper, 
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
#--------------------------------------------------------------------------------------------

import torch
from gpytorch.priors import LKJCholeskyFactorPrior, MultivariateNormalPrior, LKJCovariancePrior

class MultivariateNormalPriorPyro(MultivariateNormalPrior):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=False, transform=None):
        super().__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args, transform)
    
    def expand(self,size: torch.Size, _instance = None):
        return self

class LKJCholeskyFactorPriorPyro(LKJCholeskyFactorPrior):
    def __init__(self, n, eta, validate_args=False, transform=None):
        super().__init__(n, eta, validate_args, transform)
    
    def expand(self, size: torch.Size, _instance = None):
        return self
    
class LKJCovariancePriorPyro(LKJCovariancePrior):
    def __init__(self, n, eta, sd_prior, validate_args=False):
        super().__init__(n, eta, sd_prior, validate_args)
    
    def expand(self, batch_shape, _instance=None):
        return self