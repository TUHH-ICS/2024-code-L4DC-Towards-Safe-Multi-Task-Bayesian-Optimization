%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Author(s): Jannis Lübsen
%--------------------------------------------------------------------------------------------
% Naive gradient descent implementation. Not recommended to be used.

function X = gradient_descent(X,f,opts,varargin)
oldOpts.showIts = 0;
if isempty(opts) || ~isstruct(opts)
    opts = oldOpts;
else
    opts=getopts(oldOpts,opts);
end

if length(varargin) > 7
    error("To many inputs")
elseif length(varargin) < 7
    lp.cov = 0.01;
    lp.lik = 0.01;
    len = length(varargin);
else
    lp = varargin{end};
    len = length(varargin)-1;
end
i = 0;
disp("Iterations        Gradients           lp")
[nlZ, dnlZ] = feval(f,X, varargin{1:len});
while ((any(abs(dnlZ.cov) > 0.01) || any(abs(dnlZ.lik) > 0.01))) && i < 2000

    X.cov = X.cov - min(lp.cov,0.1/(min(abs(dnlZ.cov)))) * dnlZ.cov;
    X.lik = X.lik - min(lp.lik,0.1/(min(abs(dnlZ.lik)))) * dnlZ.lik;

    i = i + 1;
    if opts.showIts
        disp(num2str(i)+"       "+num2str(dnlZ.cov')+" "+num2str(dnlZ.lik')+ "    "+num2str(lp.cov)+" "+num2str(lp.lik))
    end
    dnlZ_old = dnlZ;
    [nlZ, dnlZ] = feval(f,X, varargin{1:len});
    if isstruct(lp)
        lp.cov=change_lp(lp.cov,dnlZ.cov,dnlZ_old.cov);
        lp.lik=change_lp(lp.lik,dnlZ.lik,dnlZ_old.lik);
    else
        lp=change_lp(lp,dnlZ.cov,dnlZ_old.cov);
    end
end
end

function lp = change_lp(lp,dnlZ_old, dnlZ)
if any((dnlZ_old<0)~=(dnlZ<0))
    lp = lp/2;
else
    return
end
end

