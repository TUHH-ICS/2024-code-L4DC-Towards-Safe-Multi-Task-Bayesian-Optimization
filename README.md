# Safe Multi-Task Bayesian Optimization



## General

This repository contains supplementary material and the code to reproduce the tables and figures presented in 

> J. O. Lübsen, C. Hespe, A. Eichler, "Safe Multi-Task Bayesian Optimization", submitted to the 6th Conference of Learning for dynamics and control, 2024

The preprint with supplementary material including proofs is available on arXiv:
http://arxiv.org/abs/2312.07281

The code has three main entry points, which are located in the `code` directory. Usually, the user needs to do some adjustments which are specified in the respective file.

1. `test_run.py` starts the optimization of a low dimensional problem with additional online generated figures. It provides a good illustration of how the algorithm works. In the file, the user can switch between a one and two dimensional problem.
2. `run_N2.py` starts to generate the data that is used in Figure 3 (a). The script needs to be executed repetitively with different disturbances.
3. `run_N5.py` starts to generate the data similar to Figure 3 (b).

Note that running scripts `run_N2.py` and `run_N5.py` may take a long time.
After generating the data, the script `plot.py` can be used to plot a figure similar to Figure 3.


## Prerequisites

To run the code install python3.10 and the dependencies specified in `requirements.txt`.

> pip install -r requirements.txt

The code in this repository was tested in the following environment:

* *Ubuntu 22.04.3 LTS
* *Python 3.10.12





