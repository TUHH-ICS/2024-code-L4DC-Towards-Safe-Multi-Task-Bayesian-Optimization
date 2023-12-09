%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Author(s): Jannis Lübsen
%--------------------------------------------------------------------------------------------
% Coordinate transformation for higher numerical stability of descent oracle

function [x] = backwardCoordTransf(cond,x_transf, varargin)
sort_abs = 0;
if nargin >= 3, sort_abs = varargin{1}; end
if nargin >=4, interval = varargin{2}; end
if nargin >= 5, error("two man input arguments"); end
if nargin < 4, interval = [-1,1]; end
if size(cond,1) ~= size(x_transf,2), error("dimension of parameters and boundaries does not match"); end
if any(cond(:,2)-cond(:,1) < 0), error("cond is wrong arranged:cond = [lower, upper]"); end

b = interval(2);
a = interval(1);
di = b-a;
slope_ = (cond(:,2)'-cond(:,1)')/di;
if sort_abs
    cond = sort(cond,2,"ascend","ComparisonMethod","abs");
end
x = slope_.*(x_transf-b)+cond(:,2)';
end
