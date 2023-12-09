import torch
from botorch.models import MultiTaskGP
from botorch.utils import draw_sobol_samples
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from model.model import MultiTaskGPICM
from cov.task_cov import IndexKernelAllPriors
from gpytorch.priors import UniformPrior, MultivariateNormalPrior, GammaPrior, LogNormalPrior
from gpytorch.constraints import Interval, Positive
from gpytorch.means import ConstantMean
from utils.priors import LKJCholeskyFactorPriorPyro


lb1 = .05
ub1 = .2
lb2 = 1.e1
ub2 = 2.e1

def build_gp(train_inputs, train_targets):
    num_tasks = train_inputs[-1].to(dtype=torch.int64).max() + 1
    d = train_inputs[0].shape[-1]
    train_inputs = torch.hstack(train_inputs).squeeze()
    likelihood = GaussianLikelihood(
                    #UniformPrior(1e-2, 3.e-1),
                    #Interval(1e-2, 3.e-1)
                 )
    lbl = torch.tensor([lb1, lb2]).repeat(1,int(d/2))
    ubl = torch.tensor([ub1, ub2]).repeat(1,int(d/2))
    gp = MultiTaskGPICM(
        train_inputs,
        train_targets,
        task_feature=-1,
        likelihood=likelihood,
        covar_module=ScaleKernel(
            RBFKernel(ard_num_dims=d,
                #lengthscale_prior=UniformPrior(lbl, ubl),
                #lengthscale_constraint=Interval(lbl, ubl),
            ),
            # outputscale_prior=UniformPrior(10.**2, 30.**2),
            # outputscale_constraint=Interval(10.**2, 30.**2),
            #outputscale_prior=UniformPrior(1**2*0.99, 30**2*1.00),
            outputscale_constraint=Interval(0.1, 10**2),
        ),
        task_covar_module=IndexKernelAllPriors(
            num_tasks,
            num_tasks,
            covar_factor_prior=LKJCholeskyFactorPriorPyro(num_tasks,10)
                            ),

        mean_module=ConstantMean()
    )
    #gp.task_covar_module.covar_factor.requires_grad=False
    # gp.task_covar_module._set_covar_factor(torch.eye(num_tasks))
    gp.mean_module._constant_closure(gp.mean_module,torch.tensor([0.5]))
    gp.covar_module._set_outputscale(torch.tensor(1))
    gp.likelihood.noise = torch.tensor([0.001])
    gp.covar_module.base_kernel._set_lengthscale(torch.tensor([[.4]]))
    #gp.mean_module.constant.requires_grad=False
    #gp.likelihood.raw_noise.requires_grad = False
    #gp.covar_module.base_kernel.raw_lengthscale.requires_grad = False
    #gp.covar_module.raw_outputscale.requires_grad = False
    return gp

def build_gp2(train_inputs, train_targets):
    num_tasks = train_inputs[-1].to(dtype=torch.int64).max() + 1
    d = train_inputs[0].shape[-1]
    train_inputs = torch.hstack(train_inputs).squeeze()
    likelihood = GaussianLikelihood(
                    #UniformPrior(1e-2, 3.e-1),
                    #Interval(1e-2, 3.e-1)
                 )
    lbl = torch.tensor([lb1, lb2]).repeat(1,int(d/2))
    ubl = torch.tensor([ub1, ub2]).repeat(1,int(d/2))
    gp = MultiTaskGPICM(
        train_inputs,
        train_targets,
        task_feature=-1,
        likelihood=likelihood,
        covar_module=ScaleKernel(
            RBFKernel(ard_num_dims=d,
                #lengthscale_prior=UniformPrior(lbl, ubl),
                #lengthscale_constraint=Interval(lbl, ubl),
            ),
            # outputscale_prior=UniformPrior(10.**2, 30.**2),
            # outputscale_constraint=Interval(10.**2, 30.**2),
            #outputscale_prior=UniformPrior(1**2*0.99, 30**2*1.00),
            outputscale_constraint=Interval(0.1, 10**2),
        ),
        task_covar_module=IndexKernelAllPriors(
            num_tasks,
            num_tasks,
            covar_factor_prior=LKJCholeskyFactorPriorPyro(num_tasks,1e-1),
            covar_factor_constraint=Interval(torch.tensor([[1,0],[-1,-1]]),(1e-8)*torch.ones(2,2)+torch.tensor([[1,0],[1,1]]))
                            ),

        mean_module=ConstantMean()
    )
    #gp.task_covar_module.covar_factor.requires_grad=False
    # gp.task_covar_module._set_covar_factor(torch.eye(num_tasks))
    gp.mean_module._constant_closure(gp.mean_module,torch.tensor([0.5]))
    gp.covar_module._set_outputscale(torch.tensor(1))
    gp.likelihood.noise = torch.tensor([0.001])
    gp.covar_module.base_kernel._set_lengthscale(torch.tensor([[.4]]))
    #gp.mean_module.constant.requires_grad=False
    #gp.likelihood.raw_noise.requires_grad = False
    #gp.covar_module.base_kernel.raw_lengthscale.requires_grad = False
    #gp.covar_module.raw_outputscale.requires_grad = False
    return gp

def plot_post(gp, task, test_x, sqrtbeta=None, ax = None):
    with torch.no_grad():
        posterior = gp.posterior(test_x.reshape(-1, 1, 1), output_indices=task)
        ymean = posterior.mean.squeeze()
        se = posterior.variance.squeeze().sqrt()
        if sqrtbeta == None:
            sqrtbeta = 2
    u =  sqrtbeta * se + ymean
    l =  -sqrtbeta * se + ymean
    x = gp.train_inputs[0]
    i = x[:, -1]
    x = x[:, 0:-1]
    i = i.squeeze()
    y = gp.train_targets
    fill = ax.fill_between(test_x, u.squeeze(), l.squeeze(), alpha=0.2, color="b")
    mean, = ax.plot(test_x, ymean, "C3")
    obs2, = ax.plot(x[i == 1], y[i == 1], "+C0")
    x1 = x[i == 0]; y1 = y[i == 0]
    obs1, = ax.plot(x1, y1, "oC2")
    # ax.hlines(T, test_x.min(), test_x.max(), colors="C0")
    if sqrtbeta != 2:
        ax.legend([(fill,mean)],[r"$\mu(x) \pm \bar{\beta}^{1/2}\sigma(x)$"])
    else:
        ax.legend([(fill,mean)],[r"$\mu(x) \pm \beta^{1/2}\sigma(x)$"])
    return obs1,obs2


def plot_post2D(gp,task,bounds,samps=101,sqrtbeta=None):
    figure = plt.figure(1)
    figure.clf()

    sqrtbeta = 2 if sqrtbeta == None else sqrtbeta
    X, Y = _mesh_helper(bounds,samps)
    test_x = torch.cat((X.reshape(X.numel(),1), Y.reshape(Y.numel(),1)),-1)
    with torch.no_grad():
        posterior = gp.posterior(test_x, output_indices=task)
        ymean, yvar = posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)
    ax = figure.add_subplot(1,1,1,projection='3d')
    ymean = ymean.reshape(X.size())
    yvar = yvar.reshape(X.size())
    ax.plot_wireframe(X.cpu().detach().numpy(),Y.cpu().detach().numpy(),(ymean+sqrtbeta*torch.sqrt(yvar)).cpu().detach().numpy(), color='r', rstride=10, cstride=10)
    ax.plot_wireframe(X.cpu().detach().numpy(),Y.cpu().detach().numpy(),(ymean-sqrtbeta*torch.sqrt(yvar)).cpu().detach().numpy(), color='b', rstride=10, cstride=10)
    x = gp.train_inputs[0]
    i = x[:,-1]
    x = x[:,0:-1]
    i = i.squeeze()
    y = gp.train_targets
    x1 = x[i==0].cpu().detach().numpy()
    x2 = x[i==1].cpu().detach().numpy()
    ax.scatter(x1[:,0], x1[:,1] ,y[i==0].cpu().detach().numpy(),"+r")
    ax.scatter(x2[:,0], x2[:,1] ,y[i==1].cpu().detach().numpy(),"+b")
    plt.show()

def _mesh_helper(bounds,samps=101):
    x1 = torch.linspace(bounds[0,0],bounds[1,0],samps)
    x2 = torch.linspace(bounds[0,1],bounds[1,1],samps)
    return torch.meshgrid(x1, x2, indexing='ij')

def sample_from_task(obj, tasks, bounds, n=5, data = None):
    for i in tasks:
        X_init = draw_sobol_samples(bounds = bounds, n=n, q=1).reshape(n,bounds.size(-1))
        data = concat_data((X_init,
                    i*torch.ones(X_init.size(0),1),
                    obj[i](X_init)),data)
    return data

def concat_data(data, mem = None):
    if mem is None:
        return data
    else:
       return tuple([torch.vstack((mem,data)) for mem,data in zip(mem,data)])
