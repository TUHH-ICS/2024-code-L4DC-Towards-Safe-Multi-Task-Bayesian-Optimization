% BUILD_MODEL Constructs dynamic model(s) for a laser-lock setup from system parameters.
% Model time will be scaled in seconds unless changed by `time_unit`, model output will be scaled in
% 10^sys_params.scaling unless changed by `scaling`. It is assumed that all parameters in sys_params are consistent
% (e.g. sys_params.k_phi should be scaled according to sys_params.scaling). If specified, uncertain system 
% parameters are prefixed with "`id`_" to avoid naming conflicts. Returns new system structure with rescaled models.
% So far, only piezo resonance parameters may be specified as uncertain.

function sys = build_laser_model(sys_params, scaling, time_unit)
    arguments
        sys_params
        scaling (1,1) double {mustBeFinite, mustBeInteger} = sys_params.scaling
        time_unit (1,:) char {mustBeMember(time_unit, {'seconds', 'milliseconds', 'microseconds'})} = 'seconds' %mustBeTextScalar
    end
    
    sys = sys_params;
    sys.scaling = scaling;
    sys.k_phi = sys.k_phi * 10^(sys.scaling - sys_params.scaling);
    
    sys.Fd = sys_params.Fd * 10^(sys.scaling - sys_params.scaling);
    sys.Fd.InputName = 'w'; sys.Fd.OutputName = 'd'; % Should be set already, just making sure.
    
    if isfinite(sys.pzt_lp_cfreq)
    	sys.G_pzt = zpk([], -2*pi * [sys.pzt_lp_cfreq, sys.pzt_fixed_pole_cfreq], 1);
    else
    	sys.G_pzt = zpk([], -2*pi * sys.pzt_fixed_pole_cfreq, 1);
    end
    sys.G_pzt.K = sys.G_pzt.K * sys.pzt_gain / dcgain(sys.G_pzt);
    sys.G_pzt.TimeUnit = 's';
    sys.G_pzt.InputName = 'u'; sys.G_pzt.OutputName = 'U';
    
    % We just need this for G_l, so don't store in sys.
    G_pz = minreal(sys.G_pz * tf('s')); % Perform pole-zero cancelation.
    G_pz = G_pz / dcgain(G_pz); % DC gain of G_pz is absorbed in k_omega.
    
    sys.G_l = 10^sys.scaling * sys.k_omega / sys.omega_nom / tf('s') / G_pz;
    sys.G_l.TimeUnit = 's';
    sys.G_l.InputName = 'U'; sys.G_l.OutputName = 'phi';
    
    sys.G = connect(sys.G_pzt, sys.G_l, sys.Fd, sumblk('y = phi + d'), {'w', 'u'}, {'y'}); % Connect to create full model
    
    sys.C = pid(sys.kp, sys.ki * sys.ctrl_rate) / sys.k_phi;  % Continuous time equivalent to FW implementation of discrete time PI controller.
    sys.C.TimeUnit = 's';
    sys.C.InputName = 'e'; sys.C.OutputName = 'u';
    
    if ~strcmp(time_unit, 'seconds')
        sys.G = chgTimeUnit(sys.G, time_unit);
        sys.Fr = chgTimeUnit(sys.Fr, time_unit);
        sys.Fd = chgTimeUnit(sys.Fd, time_unit);
        sys.G_l = chgTimeUnit(sys.G_l, time_unit);
        sys.G_pzt = chgTimeUnit(sys.G_pzt, time_unit);
        sys.G_pz = chgTimeUnit(sys.G_pz, time_unit);
        sys.C = chgTimeUnit(sys.C, time_unit);
    end
    
    sys.CL = connect(sys.G, sys.C, sumblk('e = r - y'), {'r', 'w'}, {'y'});
    % sys.S = 1 / (1 + sys.C * sys.G);
    % sys.T = sys.C * sys.G / sys.S;
end
