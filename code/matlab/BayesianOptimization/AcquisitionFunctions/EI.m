%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Implements the expected improvement acquisition of
% Jones, D.R., Schonlau, M., and Welch, W.J. (1998).
% Efficient global optimization of expensive black-box func-tions.
% Author(s): Jannis Lübsen
%--------------------------------------------------------------------------------------------
% Expected improvement function
% xi describes the offset
% maxProb defines whether the acq function should be maximized

function [acq]=EI(x,hyp,inf_,mean_,cov_,lik_,xt,post,yt,opts,varargin)
oldOpts.xi = 0.01;
oldOpts.maxProb = 0;

opts = getopts(oldOpts,opts);

if ~isempty(varargin{1}) && strcmp(varargin{1}.name,'lineBO')
    n = size(x,1);
    AlgoStruct = varargin{1};
    l = AlgoStruct.l;
    x_vec = AlgoStruct.x_vec;
    if n > 1
        x_vec = repmat(x_vec,[n,1]);
    end
    x_vec(:,l) = x;
    x = x_vec;
end

if opts.maxProb
    f=max(yt);
else
    f=min(yt);
end
[~,~,mu,var]=gp(hyp,inf_,mean_,cov_,lik_,xt,post,x);
acq = -calcEI(mu,var,f,opts.maxProb, opts.xi);
end

function [acq] = calcEI(mu,var,f, maxProb, xi)
sig = sqrt(var);
if maxProb
    res = (mu-(f+xi));
    Z = res./sig;
else
    res = (f-xi-mu);
    Z = res./sig;
end
Z(sig==0)=0;
acq = res.*normcdf(Z)+sig.*normpdf(Z);
end
