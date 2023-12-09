%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Implements the safe options with constant beta of
% Berkenkamp,  F.,  Schoellig,  A.P.,  and  Krause,  A.  (2016).
% Safe controller optimization for quadrotors with Gaussian  processes.
% Author(s): Jannis Lübsen
%--------------------------------------------------------------------------------------------
% SafeOpt function
% Expander search is after num_expander_eval aborted

function [varargout] = safeOpt(hyp, inf_, mean_, cov_, lik_, xt, yt,post,acq_func,opts,algo_data)
safeOpts = opts.safeOpts;
oldSafeOpts.thresholdOffset = safeOpts.threshold;
oldSafeOpts.thresholdPer = 0.2;
oldSafeOpts.Optimize = 0;
oldSafeOpts.xs_safe = [];
oldSafeOpts.beta = 2;
safeOpts=getopts(oldSafeOpts,safeOpts);
num_expander_eval = 20;

w = @(x) UCB(x,hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts,algo_data) + LCB(x,hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts,algo_data);
xs_safe = safeOpts.xs_safe;
threshold = safeOpts.threshold;
thresh_per = safeOpts.thresholdPer;
thresh_offset = safeOpts.thresholdOffset;
U = -UCB(xs_safe, hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts, algo_data);
if ~isfield(safeOpts,'current_ymin') || safeOpts.current_ymin ~= min(yt) || ~isfield(safeOpts,'threshold_vec')
    safeOpts.current_ymin = min(yt);
    thresh_max = max(0,threshold-min(yt)-thresh_offset);

    safeOpts.threshold_vec = calc_threshold(thresh_max, thresh_per, xs_safe, opts.safeOpts, algo_data);
end
I_S = U <= safeOpts.threshold_vec;

S = xs_safe(I_S,:);
[min_u, I_minU] = min(U);
L = LCB(S,hyp, inf_, mean_, cov_, lik_, xt,post,yt,safeOpts,algo_data);
I_M = L < min_u;
M = S(I_M,:);
S_without_M = S(~I_M,:);
if length(algo_data.l) == 1
    G_old = S([1,end],:);
else
    min_wm = min(w(M));
    if ~isempty(S_without_M)
        I_G = w(S_without_M) < min_wm;
        G_old = S_without_M(I_G,:);
    else
        G_old = [];
    end
end
g = false(size(G_old,1),1);
if any(~I_S) && ~isempty(G_old)
    for i = 1:size(G_old,1)
        x_add = G_old(i,:);
        y_add = LCB(x_add,hyp, inf_, mean_, cov_, lik_, xt, yt,post,safeOpts,algo_data);
        if strcmp(algo_data.name, 'lineBO')
            x_vec = algo_data.x_vec;
            x_vec(algo_data.l) = x_add;
            x_add = x_vec;
        end
        xt_temp = [xt;x_add];
        yt_temp = [yt;y_add];
        [~,~,post_test] = gp(hyp, inf_, mean_, cov_, lik_,xt_temp,yt_temp);
        if any(-UCB(xs_safe(~I_S,:),hyp, inf_, mean_, cov_, lik_,xt_temp,post_test,yt_temp,safeOpts,algo_data) <= threshold)
            g(i) = true;
            if i > num_expander_eval
                break;
            end
        end
    end
    G = G_old(g,:);
else
    G=[];
end
disp("Number of tests for G: " + num2str(length(G)))

if isempty(G) && isempty(M)
    varargout{1} = zeros(1,size(xs_safe,2));
    varargout{2} = 0;
    return;
end
if ~isempty(G)
    A = union(G,M,"rows");
else
    A = M;
end
nacq = w(A) / (2 * safeOpts.beta);
if algo_data.plot && length(algo_data.l) == 1
    figure(4)
    plot(S,ones(length(S),1),'s','MarkerFaceColor','b')
    hold on
    plot(M,ones(length(M),1)-0.5,'s','MarkerFaceColor','r')
    plot(G,ones(length(G),1)*0,'s','MarkerFaceColor','y')
    xlim([xs_safe(1) xs_safe(end)])
    hold off
end

[m_nacq, I_x] = min(nacq);
varargout{2} = m_nacq ;
varargout{1} = A(I_x,:);
varargout{3} = algo_data;
end
