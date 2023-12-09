#!/usr/bin/env python3

#---------------------------------------------------------------------------------------------
# For Paper, 
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
#--------------------------------------------------------------------------------------------


from typing import List
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.module import Module
from gpytorch.priors.prior import Prior
from torch import Tensor


class MultiTaskGPICM(MultiTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        task_covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | None = None,
        output_tasks: List[int] | None = None,
        rank: int | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            task_feature,
            train_Yvar,
            mean_module,
            covar_module,
            likelihood,
            task_covar_prior,
            output_tasks,
            rank,
            input_transform,
            outcome_transform,
        )
        if task_covar_module is not None:
            self.task_covar_module = task_covar_module
