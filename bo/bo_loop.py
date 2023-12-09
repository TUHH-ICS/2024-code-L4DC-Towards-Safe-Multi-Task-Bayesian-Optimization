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
from torch import Tensor
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from utils.utils import concat_data
from botorch.acquisition.objective import ScalarizedPosteriorTransform

N_TOL = -1e-6


class BayesianOptimization:
    def __init__(
        self,
        obj,
        tasks,
        bounds,
        threshold,
        num_acq_samps: list = [1,1,1],
        boundary_T= -15.0,
    ):
        self.obj = obj
        self.bounds = bounds
        self.threshold = threshold
        self.boundary_T = boundary_T
        self.num_acq_samps = num_acq_samps
        self.tasks = tasks
        self.run = 0
        self.best_y = []
        self.best_x = []
        self.dim = bounds.size(-1)
        self.gp = None

    def step(self):
        self.run += 1
        print("Run : ", self.run)
        print(f"Best value found: {self.observed_max[0]: .3f}")
        print(f"Worst value: {self._get_min_observed()[0]}")
        W = torch.eye(len(self.tasks))
        for i in self.tasks:
            posterior_transform = ScalarizedPosteriorTransform(W[:, i].squeeze())
            new_point = self.get_next_point(i, posterior_transform)
            if i == 0:
                print(f"New Point: {new_point}")
                new_point_task0 = new_point
            if i != 0:
                new_point = torch.vstack((new_point, new_point_task0))
            new_result = self.obj[i](new_point)
            self.train_inputs, self.train_tasks, self.train_targets = concat_data(
                (new_point, i * torch.ones(new_point.shape[0], 1), new_result),
                (self.train_inputs, self.train_tasks, self.train_targets),
            )
        self.observed_max = self._get_max_observed()
        self.best_y.append(self.observed_max[0])
        self.best_x.append(self._get_best_input()[0])
        return self.train_inputs, self.train_tasks, self.train_targets

    def inequality_consts(self, input: Tensor):
        self.gp.eval()
        inputx = input.view(int(input.numel() / self.dim), self.dim)
        output = self.gp(torch.hstack((inputx, torch.zeros(inputx.size(0), 1))))
        val = (
            output.mean
            - output.covariance_matrix.diag().sqrt() * self.sqrtbeta
            - self._det_threshold(inputx)
        )

        return val.view(inputx.shape[0], 1)

    def update_gp(self, gp, sqrtbeta):
        with torch.no_grad():
            self.train_inputs = gp.train_inputs[0][..., :-1]
            self.train_tasks = gp.train_inputs[0][..., -1:]
            self.train_targets = gp.train_targets.unsqueeze(-1)
            self.sqrtbeta = sqrtbeta.detach()
        if self.gp is None:
            self.observed_max = self._get_max_observed()
            self.best_y.append(self.observed_max[0])
            self.best_x.append(self._get_best_input()[0])
        self.gp = gp

    def _random_walk(self, initial_condition, maxiter=20, init_step_size=0.005):
        step_size = init_step_size
        cond_old = initial_condition
        for i in range(maxiter):
            cond_new = step_size * torch.randn(cond_old.size()) + cond_old
            out_upper = cond_new > 1 - N_TOL
            out_lower = cond_new < N_TOL
            if out_upper.any():
                cond_new[out_upper] = 1.0
            if out_lower.any():
                cond_new[out_lower] = 0.0
            conds_ff = (self.inequality_consts(cond_new) >= 0.0).squeeze()
            if conds_ff.all():
                cond_old = cond_new
                step_size *= 2
            elif conds_ff.any():
                cond_old[conds_ff, ...] = cond_new[conds_ff, ...]
                step_size /= 2
            else:
                step_size /= 10
        return cond_old.view(*initial_condition.shape)

    def _det_threshold(self, x):
        min_dist = torch.tensor([0.1])
        inf = float("inf")
        lower = torch.linalg.norm(x.squeeze(0) - self.bounds[0, :], dim=-1, ord=-inf)
        upper = torch.linalg.norm(x.squeeze(0) - self.bounds[1, :], dim=-1, ord=-inf)
        new_threshold = self.boundary_T + torch.minimum(lower, upper).minimum(
            min_dist
        ) / min_dist * (self.threshold - self.boundary_T)
        return new_threshold.view(new_threshold.numel())

    def _get_max_observed(self):
        return [
            torch.max(self.train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]

    def _get_min_observed(self):
        return [
            torch.min(self.train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]

    def _get_best_input(self):
        return [
            self.train_inputs[self.train_tasks.squeeze() == i, ...][
                torch.argmax(self.train_targets[self.train_tasks == i])
            ]
            for i in self.tasks
        ]

    def _get_initial_cond(self):
        _, ind = self.train_targets[self.train_tasks.squeeze() == 0].sort(
            dim=0, descending=True
        )
        sorted_train_inp = self.train_inputs[self.train_tasks.squeeze() == 0][ind, ...]
        eqfull = self.inequality_consts(sorted_train_inp).squeeze()
        pot_cond = sorted_train_inp.view(
            self.train_inputs[self.train_tasks.squeeze() == 0].size()
        )[eqfull >= 0, ...][:5, ...]
        return pot_cond.view(pot_cond.size(0), 1, self.dim)

    def get_next_point(self, task, posterior_transform):
        if task == 0:
            init_cond = self._random_walk(self._get_initial_cond())
            if self.run >= 4 and self.run % 3 != 0:
                acq = qExpectedImprovement(
                    self.gp,
                    self.observed_max[task],
                    posterior_transform=posterior_transform,
                )
            else:
                return init_cond[torch.randint(init_cond.size(0), (1,)), ...].view(
                    1, self.dim
                )
        else:
            if self.run % 2 == 0:
                acq = qExpectedImprovement(
                    self.gp,
                    self.observed_max[task],
                    posterior_transform=posterior_transform,
                )
            else:
                acq = qUpperConfidenceBound(
                    self.gp,
                    beta=10 * self.sqrtbeta,
                    posterior_transform=posterior_transform,
                )
        candidate, tt = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=self.num_acq_samps[task],
            num_restarts=init_cond.size(0) if task == 0 else 1,
            raw_samples=512 if task != 0 else None,
            nonlinear_inequality_constraints=[self.inequality_consts]
            if task == 0
            else None,
            batch_initial_conditions=init_cond if task == 0 else None,
            options={"maxiter": 20},
        )
        return candidate
