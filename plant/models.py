#!/usr/bin/env python3

#---------------------------------------------------------------------------------------------
# For Paper, 
# "Safe Multi-Task Bayesian Optimization"
# by Jannis O. Lübsen, Christian Hespe, Annika Eichler
# Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
# Licensed under the GPLv3. See LICENSE in the project root for license information.
# Author(s): Jannis Lübsen
#--------------------------------------------------------------------------------------------


import control
import numpy as np


def get_disturbance_filter(disturbance: float | None = None):
    A = np.array(
        [
            [-1.683633670520548, -3.338955004290262],
            [3.338955004290134, -2.555004366329481e03],
        ]
    )
    B = np.array([[-55.7272209062623], [55.2229036788730]])
    C = np.array([[-55.7272209062623, -55.2229036788729]])
    D = np.array([[0.0]])
    if disturbance is not None:
        Ad = (np.random.rand(*A.shape) - 0.5) * disturbance * A
        Bd = (np.random.rand(*B.shape) - 0.5) * disturbance * B
        Cd = (np.random.rand(*C.shape) - 0.5) * disturbance * C
        Fd = control.ss(A + Ad, B + Bd, C + Cd, D)
    else:
        Fd = control.ss(A, B, C, D)
    Fd.input_labels = "w"
    Fd.output_labels = "d"
    Fd.name = "Fd"
    return Fd


def get_laser_model(disturbance: float | None = None):
    A = np.array(
        [
            [0, 0, 0, 0, 0],
            [
                0,
                -49360.8246911984,
                37414.5002321518,
                -624.819686192088,
                -84.1572490155231,
            ],
            [
                0,
                -37414.5002321517,
                -166748.568607472,
                5791.19951595552,
                781.548777769087,
            ],
            [
                0,
                624.819686196785,
                5791.19951596256,
                -53311.6935192129,
                -2714332.54856702,
            ],
            [
                0,
                -84.1572489847785,
                -781.548777753524,
                2714332.54856702,
                -972.443656861287,
            ],
        ]
    )
    B = np.array(
        [
            [5749.94849478096],
            [-50538.2017972732],
            [-18401.7601846436],
            [319.849723994729],
            [-43.0838925128083],
        ]
    )
    C = np.array(
        [
            385287.732124115,
            50538.2017972732,
            -18401.7601846436,
            319.849723995663,
            43.0838925126591,
        ]
    )
    D = np.array([[0.0]])
    if disturbance is not None:
        Ad = (np.random.rand(*A.shape) - 0.5) * disturbance * A
        Bd = (np.random.rand(*B.shape) - 0.5) * disturbance * B
        Cd = (np.random.rand(*C.shape) - 0.5) * disturbance * C
        G = control.ss(A + Ad, B + Bd, C + Cd, D)
    else:
        G = control.ss(A, B, C, D)

    # G = control.ss(A, B, C, D)
    G.input_labels = "u"
    G.output_labels = "phi"
    G.name = "G"
    return G


def get_reference_filter(disturbance: float | None = None):
    A = np.array(
        [
            [-188.775380615703, -195.546763514755, -272.645444309218],
            [-195.546763514755, -495.821433192008, -1487.53759561648],
            [272.645444309218, 1487.53759561648, -3590.80318619229],
        ]
    )
    B = np.array([[-35.4529653809644], [-20.7593434169242], [24.9362283427276]])
    C = np.array([[-35.4529653809644, -20.7593434169241, -24.9362283427276]])
    D = np.array([[0]])
    if disturbance is not None:
        Ad = (np.random.rand(*A.shape) - 0.5) * disturbance * A
        Bd = (np.random.rand(*B.shape) - 0.5) * disturbance * B
        Cd = (np.random.rand(*C.shape) - 0.5) * disturbance * C
        Fr = control.ss(A + Ad, B + Bd, C + Cd, D)
    else:
        Fr = control.ss(A, B, C, D)
    Fr.input_labels = "wr"
    Fr.output_labels = "r"
    Fr.name = "Fr"
    return Fr
