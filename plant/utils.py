#!/usr/bin/env python3

# ---------------------------------------------------------------------------------------------
# For Paper,
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
# --------------------------------------------------------------------------------------------


from control import gram, summing_junction, interconnect, tf2ss, tf
from control import StateSpace
from numpy import trace, sqrt, minimum
from plant.models import get_disturbance_filter, get_laser_model, get_reference_filter
from botorch.utils.transforms import unnormalize
from slycot.exceptions import SlycotError
import torch


def get_nh2(param, G, bounds, K_typ="PI"):
    param = unnormalize(param, bounds).round(decimals=7)
    num_evals = param.size(0)
    vals = torch.zeros(num_evals, 1)
    for i in range(num_evals):
        flag = False
        while not flag:
            try:
                vals[i] = torch.tensor(
                    h2_norm(get_closed_loop(param[i, ...].detach().numpy(), G, K_typ))
                )
                flag = True
            except SlycotError:
                flag = False
                print("\nSLYCOT ERROR\n")
                param[i, ...] += 1e-8 * torch.ones_like(param[i, ...])
    return -vals


max_h2 = 40.0


def h2_norm(ss: StateSpace):
    B = ss.B
    try:
        Wo = gram(ss, "o")
    except ValueError as e:
        print(f"Got error {e}.\n Returned max H2 value:{max_h2}")
        return max_h2
    H2 = minimum(sqrt(trace(B.T @ Wo @ B)), max_h2)
    return H2


def build_laser_model(num_laser: int = 1, disturbance: float | None = None):
    Fr = get_reference_filter(disturbance)
    sumblk = []
    G_list = []
    Fd_list = []
    for i in range(num_laser):
        Fd = get_disturbance_filter(disturbance)
        G = get_laser_model()
        Fd.input_labels = f"w({i})"
        Fd.output_labels = f"d({i})"
        Fd.name = f"Fd({i})"
        G.input_labels = f"u({i})"
        G.output_labels = f"phi({i})"
        G.name = f"G({i})"
        if i == 0:
            sumblk.extend(
                [
                    summing_junction(inputs=["phi(0)", "d(0)"], output=f"y(0)"),
                    summing_junction(inputs=["r", "-y(0)"], output=f"e(0)"),
                ]
            )
            Fd_list.append(Fd)
            G_list.append(G)
        else:
            Fd_list.append(Fd)
            G_list.append(G)
            sumblk.extend(
                [
                    summing_junction(inputs=[f"phi({i})", f"d({i})"], output=f"y({i})"),
                    summing_junction(
                        inputs=[f"y({i-1})", f"-y({i})"], output=f"e({i})"
                    ),
                ]
            )

    inputs = (
        [f"u({i})" for i in range(num_laser)]
        + ["r"]
        + [f"w({i})" for i in range(num_laser)]
    )
    outputs = [f"e({i})" for i in range(num_laser)] + [f"y({num_laser-1})"]
    Glaser = interconnect(sumblk + G_list + Fd_list, inputs=inputs, outputs=outputs)

    inputs[num_laser] = "wr"
    outputs[-1] = "z"
    sumblk = summing_junction(inputs=[f"-y({num_laser-1})", "r"], outputs="z")
    GlaserChain = interconnect([Glaser, sumblk, Fr], inputs=inputs, outputs=outputs)
    return GlaserChain


k_phi = 330000


def pi_controller(params):
    s = tf("s")
    num_c = int(params.shape[-1] / 2)
    params = params.reshape(num_c, 2) / k_phi
    C = []
    c = 0
    for i in params:
        Ct = tf2ss(i[0] + i[1] / s)
        Ct.name = f"C({c})"
        Ct.input_labels = f"e({c})"
        Ct.output_labels = f"u({c})"
        C.append(Ct)
        c += 1
    return C


def p_controller(params):
    num_c = params.shape[-1]
    params = params.reshape(num_c, 1) / k_phi
    C = []
    c = 0
    for i in params:
        Ct = tf(i[0], [1.0])
        Ct.name = f"C({c})"
        Ct.input_labels = f"e({c})"
        Ct.output_labels = f"u({c})"
        C.append(Ct)
        c += 1
    return C


def get_closed_loop(params, G, C="PI"):
    if C == "PI":
        K = pi_controller(params)
    elif C == "P":
        K = p_controller(params)
    num_c = len(K)
    inputs = [f"w({i})" for i in range(num_c)] + ["wr"]
    return interconnect(K + [G], inputs=inputs, outputs=["z"])
