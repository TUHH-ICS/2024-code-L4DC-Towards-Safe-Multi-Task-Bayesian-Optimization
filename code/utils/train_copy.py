#!/usr/bin/env python3

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from utils.utils2 import build_gp2

max_trials = 10


def multistart_optimization(gp, num_restarts=1, mode=2, max_iter=20):
    training_inputs = gp.train_inputs[0]
    training_targets = gp.train_targets
    gp_vec = []
    loss_vec = torch.ones(num_restarts) * 10000
    for j in range(num_restarts):
        flag = False
        c = 0
        while not flag:
            if c == max_trials:
                raise TimeoutError("GP is not optimizable with mode {mode}...")
            gp = build_gp2(
                (training_inputs[:, :-1], training_inputs[:, -1:]),
                training_targets.unsqueeze(-1),
            )
            gp, loss_vec[j], flag = singlestart_optimization(
                gp, training_inputs, training_targets, mode=mode, max_iter=max_iter
            )
            c += 1
        gp_vec.append(gp)
    Id = torch.argmin(loss_vec)
    gp = gp_vec[Id]
    return gp


def singlestart_optimization(
    gp, training_inputs, training_targets, mode=2, max_iter=20
):
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp.train()
    losses = []
    if mode == 1:
        optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
        for i in range(max_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from gp
            output = gp(training_inputs)
            # Calc loss and backprop gradients
            loss = -mll(output, training_targets)
            losses.append(loss.item())
            # if loss_vec[j] > loss.item(): loss_vec[j] = loss.item()
            print(f"{loss.item():.4f}")
            loss.backward()
            optimizer.step()
    else:
        optimizer = torch.optim.LBFGS(
            gp.parameters(),
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            # tolerance_grad=1e-2,
        )

        def closure():
            optimizer.zero_grad()
            output = gp(training_inputs)
            loss = -mll(output, training_targets)
            losses.append(loss.item())
            print(f"{loss.item():.4f}")
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except:
            Warning("Optimization failed")
            return None, 0, False
    return gp, losses[-1], True
