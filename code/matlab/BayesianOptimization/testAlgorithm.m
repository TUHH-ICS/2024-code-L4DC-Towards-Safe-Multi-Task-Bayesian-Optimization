%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Author(s): Jannis Lübsen, Maximilian Schütte
%--------------------------------------------------------------------------------------------

%yalmip('clear');
clear all; close all;
%% Configuration
N = 3;  % Number of repetitions of G in system chain
N_L = 2;
%% Build model from parameters
lbsync = load('lbsync.mat');  % Load parameter database
tunit = 'seconds'; tunitexp = 0;  % Overwrite time scale (to evaluate effect on optimization problem)
scaling = 15;  % Exponent of the model output scaling, 0 = s, 12 = ps, 15 = fs, etc.
sys = build_laser_model(lbsync.sim.laser.origami, scaling, tunit);

% Plant
ctrl_gain = 1;
G = balreal(ss(series(sys.G_pzt, sys.G_l) / ctrl_gain));
G.u = 'u';
G.y = 'phi';
% Reference noise coloring filter
Fr = sys.Fr;
%Fr.P{1}(1) = -1e1 * 2*pi * 10^(-tunitexp);  % Change integral behaviour to frequency region of interest
Fr = balreal(ss(Fr));  % Alt: ss, balreal, prescale
Fr.D = zeros(size(Fr.D));  % Make proper

% Plant output disturbance coloring filter
Fd = sys.Fd;
%Fd.P{1}(1) = -1e1 * 2*pi * 10^(-tunitexp);  % Change integral behaviour to frequency region of interest
Fd = balreal(ss(Fd));  % Alt: ss, balreal, prescale
Fd.D = zeros(size(Fd.D));  % Make proper

Glaser = connect(G,Fd,sumblk('y = phi + d'),{'w','u'},{'y'});
Glaser2 = sys.G;

% Link model
if N_L > 0
    sys_link = lbsync.sim.link.short;
    sys_link.Fd.P{1}(end) = -1e-1 * 2*pi;
    G_pz = zpk(sys_link.G_pz);
    G_pz.P{1} = G_pz.P{1}(1:2);
    sys_link.G_pz = ss(G_pz);
    clear G_pz;

    sys_link = build_link_model(sys_link, scaling, tunit);
    Glink = sys_link.Gpade;
    %Glink = repmat({Glink}, 1, N_L);
end

% Connectivity
Fr.u = 'w(1)';
Fr.y = 'r';

if N_L < 1
    G = repmat({Glaser}, 1, N);
    sums = cell(1, N);

    for i = 1:N
        G{i}.u = {sprintf('w(%d)', i+1);sprintf('u(%d)', i)};
        G{i}.y = sprintf('y(%d)', i);
        if i == 1
            sums{i} = sumblk('e(1) = r - y(1)');
        else
            sums{i} = sumblk(sprintf('e(%1$d) = y(%2$d) - y(%1$d)', i, i-1));
        end
    end
    sums{end+1} = sumblk(sprintf('z = r - y(%d)', N));
else
    G = cell(1, N + N_L);
    G{1} = Glaser;
    for i=2:2:N+N_L
        G{i} = Glink;
        G{i+1} = Glaser;
    end
    for i = 1:N+N_L
        if mod(i,2) ~= 0
            G{i}.u = {sprintf('w(%d)', i+1) ;sprintf('u(%d)', i)};
            G{i}.y = sprintf('y(%d)', i);
            if i == 1
                sums{i} = sumblk('e(1) = r - y(1)');
            else
                sums{i} = sumblk(sprintf('e(%1$d) = y(%2$d) - y(%1$d)', i, i-1));
            end
        else
            G{i}.u = {sprintf('y(%d)',i-1); sprintf('w(%d)', i+1); sprintf('u(%d)', i)};
            G{i}.y = {sprintf('l(%d)', i/2);sprintf('y(%d)', i)};
            sums{i} = sumblk(sprintf('e(%d) = y(%d) - l(%d)', i, i-1, i/2));
        end
    end
    sums{end+1} = sumblk(sprintf('z = r - y(%d)', N+N_L));
end
Gg = connect(G{:}, Fr, sums{:}, {'u','w'},{'e','z'});

%%
algo = 1; % 1: LineBO + MoSaOpt, 2: LineBO + SafeOpt, 3,4: PlaneBO + ...
switch algo
    case 1
        term_acq = 0.05;
        searchCond = 6;
        K = 40;
        mosOpt = 1;
        subspaceDim = 1;
        dim_combinations = [];
    case 2
        term_acq = 1;
        searchCond = 6;
        K = 40;
        mosOpt = 0;
        subspaceDim = 1;
        dim_combinations = [];
    case 3
        term_acq = 0.4;
        searchCond = 8;
        K = 20;
        mosOpt = 1;
        subspaceDim = 2;
        dim_combinations = [1,2;3,4;5,6;7,8;9,10;1,5;1,9;5,9];
    case 4
        term_acq = 3;
        searchCond = 8;
        K = 20;
        mosOpt = 0;
        subspaceDim = 2;
        dim_combinations = [1,2;3,4;5,6;7,8;9,10;1,5;1,9;5,9];
end



Kp_max = 3e1;
Kp_min = 0.2;
Ki_max = 6e1;
Ki_min = 0;

cond_t=[Kp_min, Kp_max;
    Ki_min, Ki_max;
    0, 0.000105*350;
    0, 3;
    Kp_min, Kp_max;
    Ki_min, Ki_max;
    0, 0.000105*350;
    0, 3;
    Kp_min, Kp_max;
    Ki_min, Ki_max];
cond = repmat([-1,1],size(cond_t,1),1);


if N_L == 0
    scale = [1/5; 1/2];
    scale = repmat(scale,[N+N_L,1]);
else
    scale =[];
    scale1 = [1/5; 1/2];
    scale2 = [1/5; 1/5];
    for i = 1:N+N_L
        if mod(i,2)
            scale=[scale;scale1];
        else
            scale=[scale;scale2];
        end
    end
end


inf_ = {@infGaussLik};
mean_ = {@meanConst};
lik_ = {@likGauss};
cov_ = {@covMaternard,3};
acq = {@EI};

% initial hyperparameter
hyp.lik = log(0.5);
hyp.mean = 40;
hyp.cov = log([(cond(:,2)-cond(:,1)).*scale;15]);


% Start points
X0 = [20.6963   21.5537    0.0271    1.1841   20.5658   42.2428    0.0163    0.0587   10.0596   25.4586
    11.4464   12.9611    0.0290    2.8479    9.9614   40.2759    0.0161    2.5005   23.1119   10.0352
    23.7409   19.1115    0.0196    0.2699    3.5288    8.1776    0.0249    1.4855    5.8534   29.7003
    17.3249   50.7107    0.0271    1.7580    7.5527   39.9850    0.0031    1.8779   19.8961   43.7851
    25.3767   12.5643    0.0203    1.8897    1.1533   36.8828    0.0133    0.1486   14.7892   11.5506
    3.8679   12.3297    0.0054    0.5672    1.4710   38.1119    0.0104    1.6158   20.9159   29.9470
    12.6275   12.3585    0.0348    0.2462    3.3501    8.5225    0.0061    1.8629   17.2966    3.1247
    5.4926   23.9154    0.0049    0.0927   28.1864   18.0784    0.0109    0.9988   14.1186   38.8919
    0.9518   50.5324    0.0205    2.5623   10.5668   26.7616    0.0020    0.5313   19.9517   19.8497
    24.5825    6.0133    0.0065    1.0789    1.8898   31.3131    0.0123    0.5270    6.4266   54.3092];

opts.plot = 0; % = 1 to see plots
opts.minFunc.mode = 3;
opts.minFunc.MaxFunEvals = 50;
opts.maxProb = 0;
opts.acqFunc.xi = 0.01;
opts.acqFunc.beta = 2;
opts.trainGP.acqVal = 10;

% termination conditions, change so see how they impact the convergence rate
opts.termCondAcq = term_acq;
opts.safeOpts.searchCond = searchCond;
opts.safeOpts.explorationIt = 150;
opts.maxIt = 150;
opts_lBO.maxIt = K;

opts.trainGP.It = 501;
opts.trainGP.train = 1;
opts.safeOpt = 1;
opts.safeOpts.threshold = 50;
opts.safeOpts.thresholdOffset = 12;
opts.safeOpts.thresholdPer = 0.2;
opts.safeOpts.thresholdOrder = 1;
opts.moSaOpt=mosOpt;
opts.minFunc.rndmInt.mean = [0,50];
opts.minFunc.repeat = 5;
opts_lBO.sharedGP = true;
opts_lBO.subspaceDim = subspaceDim;
opts.beta = 1;
opts_lBO.dim_combinations = dim_combinations;
% opts_lBO.oracle = 'descent';

globOpt = 12.1; % so far best
data = cell(10,11);
fun = @(params) connect_PI(params, Gg, [1/sys.k_phi 1/sys_link.k_phi],cond_t);
for i = 1:10
    x0 = forwardCoordTransf(cond_t,X0(i,:));
    tstart = tic;
    [xopt,X,Y,DIM]=lineBO(hyp,inf_,mean_,cov_,lik_,acq, fun,cond,opts,opts_lBO,x0);
    data{i,6} = toc(tstart);
    arg =find(DIM(:,1));
    DIM = DIM(arg,:);
    data{i,7} = xopt;
    data{i,8} = X;
    data{i,9} = Y;
    data{i,10} = DIM;
    data{i,11} = connect_PI2(xopt, Gg, [1/sys.k_phi 1/sys_link.k_phi],cond_t);
    y = Y{length(DIM)};
    xp = 1:length(y);
    yp = zeros(length(y),1);
    yp2 = zeros(length(y),1);
    for j=xp
        yp(j)=min(y(1:j));
        yp2(j) = max(y(1:j));
    end
    data{i,1} = length(y);
    data{i,2} = yp(end);
    data{i,3} = yp2(end);
    temp = find(yp <= (1.01)*globOpt);
    temp2 = find(yp <= (1.05)*globOpt);
    if isempty(temp)
        data{i,4} = '-';
    else
        temp = temp(1);
        data{i,4}=[temp,yp(temp)];
    end
    if isempty(temp2)
        data{i,5} = '-';
    else
        temp2 = temp2(1);
        data{i,5}=[temp2,yp(temp2)];
    end
end
dir = pwd;
int = strfind(dir,'/');
parentDir = dir(1:int(end)-1);
save(parentDir+"/data"+"/name_of_file",'data')  % define file name and path
%%
function [y] = connect_PI(pi_params, Gg, scale,cond)
pi_params=backwardCoordTransf(cond,pi_params);
N = length(pi_params)/2;
C = cell(1,N);
len_scale = length(scale);
for i=1:N
    C{i} = pid(pi_params(1,2*i-1)*scale(len_scale-mod(i,len_scale)),pi_params(1,2*i)*scale(len_scale-mod(i,len_scale)));
    C{i}.y = sprintf('u(%d)', i);
    C{i}.u = sprintf('e(%d)', i);
end
Gcl = connect(Gg,C{:}, 'w','z');
y = norm(Gcl,2)+randn(1)*0.1;           % add small noise term to preserve positive definiteness
fprintf("y = %.2f\n",y)
end





