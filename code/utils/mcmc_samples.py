#!/usr/bin/env python3

#---------------------------------------------------------------------------------------------
# For Paper, 
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
#--------------------------------------------------------------------------------------------

import gpytorch
import pyro
from copy import deepcopy
from pyro.infer.mcmc import NUTS, MCMC


def get_mcmc_samples(gp):
    train_inputs = gp.train_inputs[0]
    train_targets = gp.train_targets
    gppyro = deepcopy(gp)
    gppyro.task_covar_module.add_prior()
    gppyro.train()

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled = gppyro.pyro_sample_from_prior()
            output = sampled.likelihood(sampled(x))
            pyro.sample("obs", output, obs=y.squeeze())
        return y

    nuts_kernel = NUTS(pyro_model, jit_compile=False, max_tree_depth=3, full_mass=True)
    mcmc_run = MCMC(
        nuts_kernel, num_samples=100, warmup_steps=100, disable_progbar=False
    )
    mcmc_run.run(train_inputs, train_targets)
    gppyro.pyro_load_from_samples(mcmc_run.get_samples())
    return gppyro
