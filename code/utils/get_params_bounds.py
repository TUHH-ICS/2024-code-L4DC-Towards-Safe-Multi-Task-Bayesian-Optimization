import torch
from utils.task_gamma import get_task_gamma
from utils.get_mean_norms import get_mean_norms
from copy import deepcopy


def getboundinggp(sampmods, model0, nmc: int | None = None, delta_max: float | None = None):

    # set number of MCMC samples and delta if not available
    if nmc == None:
        #nmc = sampmods.covar_module.base_kernel.lengthscale.shape[0]
        nmc = sampmods.covar_module.outputscale.shape[0]
    if delta_max == None:
        delta_max = 0.05

    # extract input dimension from lengthscales
    d = model0.covar_module.base_kernel.lengthscale.shape[-1]
    # num_tasks = model0.train_inputs[0][-1].max() + 1

    # concatenate hyperparameter samples generated with NUTS
    outputscale = sampmods.covar_module.outputscale  # corresponds to the signal variance (sigma_f^2)
    lengthscale = sampmods.covar_module.base_kernel.lengthscale  # lengthscale (NOT logarithm or square)
    noise = sampmods.likelihood.noise  # noise variance (\sigma_n^2)

    task_cov_fac = model0.task_covar_module._eval_covar_matrix().diag().max()
    model0.task_covar_module._set_covar_factor(model0.task_covar_module.covar_factor/task_cov_fac.sqrt())

    print(model0.task_covar_module.covar_factor)
    #task_covar0 = model0.task_covar_module._eval_covar_matrix() # corresponds to the signal variance (sigma_f^2)
    # task_cov0 = model0.task_covar_module.covar_factor
    #outputscale0 = torch.minimum(model0.covar_module.outputscale*task_cov_fac,torch.tensor([30.**2])) # corresponds to the signal variance (sigma_f^2)
    outputscale0 = model0.covar_module.outputscale
    print(f"outputscale0: {outputscale0}, task_covar_fac: {task_cov_fac}")
    lengthscale0 = model0.covar_module.base_kernel.lengthscale  # lengthscale (NOT logarithm or square)
    noise0 = model0.likelihood.noise  # noise variance (\sigma_n^2)

    task_gamma, task_lambda, task_thdoubprime = get_task_gamma(model0, sampmods, delta_max)

    #outputscale *= torch.max(sqrtdiag,dim=-1)[0]**2
    #outputscale0 *= torch.max(sqrtdiag0)**2

    hyperparsamps = [[outputscale[i].reshape(1, 1), lengthscale[i].reshape(d, 1), noise[i].reshape(1, 1)]
                     for i in range(nmc)]
    hyperparsamps = [torch.cat(samps, 0).reshape(d + 2, ) for samps in hyperparsamps]

    theta0 = [outputscale0.reshape(1, 1), lengthscale0.reshape(d, 1), noise0.reshape(1, 1)]
    theta0 = torch.cat(theta0, 0).reshape(d + 2, )

    conf = 1 - delta_max
    dimpar = theta0.shape[0]

    indmax = round(len(hyperparsamps) * conf)
    inds = torch.as_tensor([torch.abs(samp - theta0).max() for samp in hyperparsamps]).argsort()[:indmax]
    sampsinregion = [hyperparsamps[ind] for ind in inds[:indmax]]
    thprim = torch.tensor([torch.tensor([samp[i] for samp in sampsinregion]).min() for i in range(dimpar)])
    thdoubprim = torch.tensor([torch.tensor([samp[i] for samp in sampsinregion]).max() for i in range(dimpar)])
    thprimnew = torch.min(thprim, theta0)
    thdoubprimnew = torch.max(thdoubprim, theta0)

    # conf = 1 - delta_max
    # indmax = round(len(outputscale) * conf)
    # inds = torch.as_tensor(torch.abs(outputscale-outputscale0)).argsort()[:indmax]
    # samps = torch.as_tensor([outputscale[i] for i in inds])
    # thprimnew = torch.minimum(samps.min(),outputscale0)
    # thdoubprimnew = torch.maximum(samps.max(),outputscale0)

    print(f"thdoubprim: {thdoubprimnew}")
    robustmodel = deepcopy(model0)

    # hypers = {'likelihood.noise_covar.noise': thdoubprimnew[-1],
    #         'covar_module.base_kernel.lengthscale' : thprimnew[1:-1],
    #         'covar_module.outputscale' : thdoubprimnew[0],
    #         'task_covar_module.covar_factor' : task_thdoubprime
    #         }
    # robustmodel.initialize(**hypers)
    
    #robustmodel.covar_module.base_kernel._set_lengthscale(thprimnew[1:-1])
    robustmodel.covar_module._set_outputscale(thdoubprimnew[0])
    robustmodel.task_covar_module._set_covar_factor(task_thdoubprime)
    #robustmodel.covar_module._set_outputscale(torch.minimum(thdoubprim,torch.tensor([600])))
    #robustmodel.likelihood.noise = thdoubprimnew[-1]

    sqrtzeta = get_mean_norms(model0,robustmodel,task_lambda)

    maxsqrtbeta = 1.414
    # gamma = torch.sqrt(torch.prod(torch.divide(thdoubprimnew[1:-1], thprimnew[1:-1])))
    # print(f"gamma length: {gamma}")
    # gamma /= torch.sqrt(torch.prod(torch.divide(thdoubprimnew[0], theta0[0])))
    # gamma *= task_gamma

    gamma = torch.sqrt(torch.prod(torch.divide(thdoubprimnew[0], theta0[0])))
    gamma *= task_gamma

    # gamma = task_gamma * (outputscale0 / thdoubprim)
    #zeta = 0.1 #(model0.train_targets**2).sum().sqrt()/thdoubprim[-1]
    #beta = torch.as_tensor(min(4 * maxsqrbeta ** 2, betabar))
    sqrtbeta = gamma*maxsqrtbeta  #+ sqrtzeta

    print(f"gamma: {gamma}"); print(f"sqrtbeta: {sqrtbeta}"); print(f"sqrtzeta: {sqrtzeta}"); print(f"Covar_Mat{task_thdoubprime@task_thdoubprime.T}")

    # Create robust bounding hyperparameters. These correspond to the minimal lengthscales
    # and maximal signal/noise variances. Referred to as theta' in the experimental sectoin of the paper
    # throbust = thprimnew  # deepcopy(thprimnew)
    # throbust[0] = thdoubprimnew[0]  # deepcopy(thdoubprimnew[0])
    # throbust[-1] = thdoubprimnew[-1]  # deepcopy(thdoubprimnew[-1])

    # # How can the model parameters reloaded?


    return robustmodel, sqrtbeta.squeeze(), gamma.squeeze()

def get_robust_gp(sampmods, model0, delta_max: float = .05):
    gamma, task_lambda, task_thdoubprime = get_task_gamma(model0, sampmods, delta_max)

    print(f"Robust Task Covar Factor: {task_thdoubprime}")
    robustmodel = deepcopy(model0)

    robustmodel.task_covar_module._set_covar_factor(task_thdoubprime)
    sqrtzeta = get_mean_norms(model0,robustmodel,task_lambda)

    maxsqrtbeta = 2

    sqrtbeta = gamma*maxsqrtbeta  #+ sqrtzeta

    print(f"sqrtbeta: {sqrtbeta}"); print(f"sqrtzeta: {sqrtzeta}"); print(f"Covar_Mat{task_thdoubprime@task_thdoubprime.T}")

    return robustmodel, sqrtbeta.squeeze(), gamma.squeeze()