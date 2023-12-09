# A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL



## General

This repository contains the simulation code to reproduce the tables and figures presented in 

> J. O. Lübsen, M. Schütte, S. Schulz, A. Eichler, "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL", submitted to the 22nd World Congress of the International Federation of Automatic Control, 2023

It may be used as an safe Bayesian Optimization framework to optimize arbitarily functions or to recreate and validate the figures from the paper.

The MATLAB file 'lineBayesOptSim.m' contains comments that provide advice on how to use this framework to optimize user specific problems.
To reproduce the figures, run the MATLAB file 'testAlgorithm.m' and use the settings given in the file. The data is saved in the 'data' folder (We assume pwd is the matlab folder). Repeat this step for all algorithms. Then, go into the plot folder, change the name of the input string the newly generated data name and run the code. The plot may be slightly different because the procedure is not deterministic.

## Prerequisites

To run the scripts in this repository, you will need a working copy of the GPML toolbox (https://gitlab.com/hnickisch/gpml-matlab).
The code in this repository has been tested in the following environment:

* *Linux Ubuntu* 20.04.6 LTS
* *MATLAB* 2021b
* *GPML Toolbox* v4.2+dev2 



