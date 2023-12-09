%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Author(s): Jannis Lübsen
%--------------------------------------------------------------------------------------------
% Coordinate transformation for higher numerical stability of descent oracle

function [x_transf]=forwardCoordTransf(cond,x,varargin)
sort_abs = 0;
if nargin >= 3, sort_abs = varargin{1}; end
if nargin >=4, interval = varargin{2}; else, interval =[-1,1]; end
if nargin >= 5, error("two man input arguments"); end
if size(cond,1) ~= size(x,2), error("dimension of parameters and boundaries does not match"); end

b = interval(2);
a = interval(1);
di = b-a;
if sort_abs
    cond = sort(cond,2,"ascend","ComparisonMethod","abs");
end
slope_ = (cond(:,2)'-cond(:,1)')/di;
x_transf = (x-cond(:,2)')./slope_+b;
end

