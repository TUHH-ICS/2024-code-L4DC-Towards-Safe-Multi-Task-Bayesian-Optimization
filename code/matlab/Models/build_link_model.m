%build_link_model Constructs dynamic stabilized fiber transmission link model from system parameters.
% Model time unit is seconds unless changed by `time_unit`, model output will be scaled in
% 10^sys_params.scaling unless changed by `scaling`. It is assumed that all parameters in sys_params are consistent
% (e.g. sys_params.k_phi should be scaled according to sys_params.scaling). Returns new system structure with 
% rescaled models.

function sys = build_link_model(sys_params, scaling, time_unit)
    arguments
        sys_params
        scaling (1,1) double {mustBeFinite, mustBeInteger} = sys_params.scaling
        time_unit (1,:) char {mustBeMember(time_unit, {'seconds', 'milliseconds', 'microseconds'})} = 'seconds'
    end
    
    sys = sys_params;  % Make copy of system

    % Set default parameters, if not provided
    if ~isfield(sys, 'pade_order'), sys.pade_order = 2; end
    if ~isfield(sys, 'refractive_index'), sys.refractive_index = 1.47; end
    
    % Set new scaling
    sys.scaling = scaling;
    sys.k_phi = sys.k_phi * 10^(sys.scaling - sys_params.scaling);
    sys.k_act = sys.k_act * 10^(sys.scaling - sys_params.scaling);
    sys.Fd = sys_params.Fd * 10^(sys.scaling - sys_params.scaling);
    
    sys.Fd.InputName = 'w'; sys.Fd.OutputName = 'd';

    if isfinite(sys.pzt_lp_cfreq)
    	sys.G_pzt = zpk([], -2*pi * [sys.pzt_lp_cfreq, sys.pzt_fixed_pole_cfreq], 1);
    else
    	sys.G_pzt = zpk([], -2*pi * sys.pzt_fixed_pole_cfreq, 1);
    end
    sys.G_pzt.K = sys.G_pzt.K * sys.pzt_gain / dcgain(sys.G_pzt);  % Set DC-Gain to sys.pzt_gain
    sys.G_pzt.InputName = 'u'; sys.G_pzt.OutputName = 'U';
    sys.G_pzt.TimeUnit = 's'; sys.G_pzt.InputUnit = 'V'; sys.G_pzt.OutputUnit = 'V';

    sys.G_l = sys.k_act * (sys.G_pz / dcgain(sys.G_pz));  % DC gain of G_pz is absorbed in k_act
    sys.G_l.TimeUnit = 's'; sys.G_l.InputUnit = 'V';
    
    sys.G_lsu = minreal(series(sys.G_pzt, sys.G_l));
    sys.G_lsu.InputName = 'u'; sys.G_lsu.OutputName = 'l';
    
    sys.delay = sys.length * sys.refractive_index / 3e8;
    
    Fd = sys.Fd/2;
    Fd.InputName = 'w';
    Fd.OutputName = 'd';
    Fr1 = tf(1, 1, 'OutputDelay', 2*sys.delay, 'InputName', 'r', 'OutputName', 'r_dly');
    Fr2 = tf(1, 1, 'OutputDelay', sys.delay, 'InputName', 'r', 'OutputName', 'r_dly2');
    Fdelay1 = tf(1, 1, 'OutputDelay', 2*sys.delay, 'InputName', 'l_', 'OutputName', 'l_dly');
    Fdelay2 = tf(1, 1, 'OutputDelay', sys.delay, 'InputName', 'l_', 'OutputName', 'l_dly2');

    
    sys.G = sys.G_lsu / 2;
    sys.G.InputName = 'u';
    sys.G.OutputName = 'l_';
    sums{1} = sumblk('l = l_ + l_dly + d + d - r_dly + r');
    sums{2} = sumblk('y = d + l_dly2 + r_dly2');
    sys.G = connect(sys.G, Fdelay1, Fd, Fr1, Fr2, Fdelay2, sums{:}, {'r','w','u'}, {'l','y'});
    
    %   sys.G = sys.G / 2;  % half from undelayed input and half from delayed
    
    sys.Gpade = pade(sys.G, sys.pade_order);
    
    sys.C = pid(sys.kp, sys.ki * sys.ctrl_rate) / sys.k_phi;  % Continuous time equivalent to FW implementation of discrete time PI controller.
    
    if ~strcmp(time_unit, 'seconds')
        sys.Fd = chgTimeUnit(sys.Fd, time_unit);
        sys.G_l = chgTimeUnit(sys.G_l, time_unit);
        sys.G_pzt = chgTimeUnit(sys.G_pzt, time_unit);
        sys.G_pz = chgTimeUnit(sys.G_pz, time_unit);
        sys.Gnom = chgTimeUnit(sys.Gnom, time_unit);
        sys.Gpade = chgTimeUnit(sys.Gpade, time_unit);
        sys.G = chgTimeUnit(sys.G, time_unit);
        sys.C = chgTimeUnit(sys.C, time_unit);
    end
    
    % sys.CL = feedback(series(sys.Cm, sys.G), 1);
    % sys.S = 1 / (1 + sys.C * sys.G);
    % sys.T = sys.C * sys.G / sys.S;
end
