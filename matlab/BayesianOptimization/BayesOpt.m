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
N = 5;  % Number of repetitions of G in system chain
N_L = 0;
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
Ki_max = 3e1;
Ki_min = 0;


if N == 5
    cond_t=[Kp_min, Kp_max;
        Ki_min, Ki_max;
        Kp_min, Kp_max;
        Ki_min, Ki_max;
        Kp_min, Kp_max;
        Ki_min, Ki_max;
        Kp_min, Kp_max;
        Ki_min, Ki_max;
        Kp_min, Kp_max;
        Ki_min, Ki_max];
else
     cond_t=[Kp_min, Kp_max;
        Ki_min, Ki_max;
        Kp_min, Kp_max;
        Ki_min, Ki_max];
end

        
cond = repmat([-1,1],size(cond_t,1),1);

if N_L == 0
    scale = [1/10; 3/1];
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
hyp.lik = log(sqrt(.1));
hyp.mean = 30;
hyp.cov = log([(cond(:,2)-cond(:,1)).*scale;15]);


% Start points
X0 = readNPY('data/X_init_vals_num5_new.npy');

opts.plot = 0; % = 1 to see plots
opts.minFunc.mode = 3;
opts.minFunc.MaxFunEvals = 50;
opts.maxProb = 0;
opts.acqFunc.xi = 0.01;
opts.acqFunc.beta = 1;
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
opts.safeOpts.threshold = 30;
opts.safeOpts.thresholdOffset = 12;
opts.safeOpts.thresholdPer = 0.1;
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
fun = @(params) connect_PI(params, Gg, [1/sys.k_phi 1/sys.k_phi],cond_t);
for i = 1:size(X0,1)
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
    data{i,11} = connect_PI(xopt, Gg, [1/sys.k_phi 1/sys.k_phi],cond_t);
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
% dir = pwd;
% int = strfind(dir,'/');
% parentDir = dir(1:int(end)-1);
save("data"+"/output_files",'data')  % define file name and path
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
y = norm(Gcl,2)+randn(1)*0.01;           % add small noise term to preserve positive definiteness
fprintf("y = %.2f\n",y)
end





