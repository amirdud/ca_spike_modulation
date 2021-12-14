import numpy as np
import channel_funs as cf
import phys_funs as pf


def compute_derivatives_four_variables_model(y, t_curr, amp, amp_syn, ts, tau1, tau2,
                                             gCa_HVAbar, gImbar, gl_pas,
                                             E_Ca, E_K, E_l, A, Cm):
    dy = np.zeros((4,))

    v         = y[0]
    Ca_HVA_m  = y[1]
    Ca_HVA_h  = y[2]
    Im_m      = y[3]

    gCa_HVA = pf.calc_conductance('HVA', gCa_HVAbar, [Ca_HVA_m, Ca_HVA_h])
    gIm = pf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = pf.calc_current(v, gCa_HVA, E_Ca)
    I_Im = pf.calc_current(v, gIm, E_K)
    I_l = pf.calc_current(v, gl_pas, E_l)
    I_I_ext = pf.I_ext_constant(amp) / A
    I_I_ext_syn = pf.I_ext_syn(t_curr, amp_syn=amp_syn, ts=ts, tau1=tau1, tau2=tau2) / A

    # dv/dt
    dy[0] = (I_I_ext + I_I_ext_syn - I_Ca_HVA - I_Im - I_l) / Cm

    # Ca_HVA dm/dt
    m_inf = cf.Ca_HVA_m_inf(v)
    m_tau = cf.Ca_HVA_m_tau(v)
    dy[1] = (m_inf - Ca_HVA_m)/m_tau

    # Ca_HVA dh/dt
    h_inf = cf.Ca_HVA_h_inf(v)
    h_tau = cf.Ca_HVA_h_tau(v)
    dy[2] = (h_inf - Ca_HVA_h)/h_tau

    # Im dm/dt
    alpha, beta = cf.Im_m_alpha_beta(v)
    dy[3] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


def compute_derivatives_two_variables_model_constant_generation(y, t_curr, amp,
                                                                gCa_HVAbar, gImbar, gl_pas,
                                                                E_Ca, E_K, E_l, A, Cm,
                                                                shift, slope, k):
    dy = np.zeros((2,))

    v         = y[0]
    Im_m      = y[1]

    Ca_HVA_h_inf = cf.Ca_HVA_h_inf(E_l)  # constant
    Ca_HVA_m_inf = cf.Ca_HVA_m_inf(v)  # instantaneous
    gCa_HVA = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])  # mS/cm2
    gIm = pf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = pf.calc_current(v, gCa_HVA, E_Ca)  # uA/cm2
    I_Im = pf.calc_current(v, gIm, E_K)
    I_l = pf.calc_current(v, gl_pas, E_l)
    I_I_ext = pf.I_ext_constant(amp) / A

    # dv/dt
    dy[0] = (I_I_ext - I_Ca_HVA - I_Im - I_l) / Cm

    # Im dm/dt
    alpha, beta = cf.Im_m_alpha_beta_with_params(v, shift, slope, k)
    dy[1] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


def compute_derivatives_two_variables_model_syn_generation(y, t_curr, amp, amp_syn, ts, tau1, tau2,
                                                           gCa_HVAbar, gImbar, gl_pas,
                                                           E_Ca, E_K, E_l, A, Cm,
                                                           shift, slope, k):
    dy = np.zeros((2,))

    v         = y[0]
    Im_m      = y[1]

    Ca_HVA_h_inf = cf.Ca_HVA_h_inf(E_l)  # constant
    Ca_HVA_m_inf = cf.Ca_HVA_m_inf(v)  # instantaneous
    gCa_HVA = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])  # mS/cm2
    gIm = pf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = pf.calc_current(v, gCa_HVA, E_Ca)  # uA/cm2
    I_Im = pf.calc_current(v, gIm, E_K)
    I_l = pf.calc_current(v, gl_pas, E_l)
    I_I_ext = pf.I_ext_constant(amp) / A
    I_I_ext_syn = pf.I_ext_syn(t_curr, amp_syn=amp_syn, ts=ts, tau1=tau1, tau2=tau2) / A

    # dv/dt
    dy[0] = (I_I_ext + I_I_ext_syn - I_Ca_HVA - I_Im - I_l) / Cm

    # Im dm/dt
    alpha, beta = cf.Im_m_alpha_beta_with_params(v, shift, slope, k)
    dy[1] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


def compute_derivatives_two_variables_model_with_perturbation(y, t_curr, amp,
                                                              amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                                              amp_syn_pert, ts_pert,tau1_pert, tau2_pert,
                                                              gCa_HVAbar, gImbar, gl_pas,
                                                              E_Ca, E_K, E_l, A, Cm,
                                                              shift, slope, k):

    dy = np.zeros((2,))

    v         = y[0]
    Im_m      = y[1]

    Ca_HVA_h_inf = cf.Ca_HVA_h_inf(E_l)  # constant
    Ca_HVA_m_inf = cf.Ca_HVA_m_inf(v)  # instantaneous
    gCa_HVA = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])  # mS/cm2

    gIm = pf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = pf.calc_current(v, gCa_HVA, E_Ca)  # uA/cm2
    I_Im = pf.calc_current(v, gIm, E_K)
    I_l = pf.calc_current(v, gl_pas, E_l)
    I_I_ext = pf.I_ext_constant(amp) / A
    I_I_ext_syn_gen = pf.I_ext_syn(t_curr, amp_syn=amp_syn_gen, ts=ts_gen, tau1=tau1_gen,tau2=tau2_gen) / A
    I_I_ext_syn_pert = pf.I_ext_syn(t_curr, amp_syn=amp_syn_pert, ts=ts_pert, tau1=tau1_pert, tau2=tau2_pert) / A

    # dv/dt
    dy[0] = (I_I_ext + I_I_ext_syn_gen + I_I_ext_syn_pert - I_Ca_HVA - I_Im - I_l) / Cm

    # Im dm/dt
    alpha, beta = cf.Im_m_alpha_beta_with_params(v, shift, slope, k)
    dy[1] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


def two_variables_model_constant_generation_calc_x_dot_all_pairs(x, amp,
                                                                 gCa_HVAbar, gImbar, gl_pas,
                                                                 E_Ca, E_K, E_l, A, Cm,
                                                                 shift, slope, k):
    x_dot = np.zeros_like(x)

    n_conditions = x.shape[0]

    for i in range(n_conditions):
        v = x[i, 0]
        Im_m = x[i, 1]

        Ca_HVA_h_inf = cf.Ca_HVA_h_inf(E_l)
        Ca_HVA_m_inf = cf.Ca_HVA_m_inf(v)

        gCa_HVA = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])

        gIm = pf.calc_conductance('Im', gImbar, [Im_m])

        I_Ca_HVA = pf.calc_current(v, gCa_HVA, E_Ca)
        I_Im = pf.calc_current(v, gIm, E_K)
        I_l = pf.calc_current(v, gl_pas, E_l)
        I_I_ext = pf.I_ext_constant(amp) / A

        # dv/dt
        v_dot = (I_I_ext - I_Ca_HVA - I_Im - I_l)/Cm

        # Im dm/dt
        alpha, beta = cf.Im_m_alpha_beta_with_params(v, shift, slope, k)
        Im_m_dot = (alpha * (1.0 - Im_m)) - (beta * Im_m)

        x_dot[i, :] = np.hstack((v_dot, Im_m_dot))
    return x_dot


def two_compartments_model_with_perturbation(y, t_curr, amp_soma, amp_nexus,
                                             amp_nexus_syn_gen, ts_nexus_gen, tau1_nexus_gen, tau2_nexus_gen,
                                             amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert, tau2_nexus_pert,
                                             gCa_HVAbar, gImbar, gl_pas, gK, gNa, gL, g_con,
                                             E_Ca, E_K, E_l, E_Na, A_soma, A_nexus, Cm_soma, Cm_nexus,
                                             shift, slope, k):

    dy = np.zeros((6,))

    Vm_soma = y[0]
    n = y[1]
    m = y[2]
    h = y[3]
    Vm_nexus = y[4]
    Im_m = y[5]

    # nexus
    Ca_HVA_h_inf = cf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf = cf.Ca_HVA_m_inf(Vm_nexus)  # inst
    gCa_HVA = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])
    gIm = pf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = pf.calc_current(Vm_nexus, gCa_HVA, E_Ca)
    I_Im = pf.calc_current(Vm_nexus, gIm, E_K)
    I_l = pf.calc_current(Vm_nexus, gl_pas, E_l)
    I_I_ext_nexus = pf.I_ext_constant(amp_nexus) / A_nexus
    I_I_ext_nexus_syn_gen = pf.I_ext_syn(t_curr, amp_syn=amp_nexus_syn_gen, ts=ts_nexus_gen,
                                         tau1=tau1_nexus_gen, tau2=tau2_nexus_gen) / A_nexus
    I_I_ext_nexus_syn_pert = pf.I_ext_syn(t_curr, amp_syn=amp_nexus_syn_pert, ts=ts_nexus_pert,
                                          tau1=tau1_nexus_pert, tau2=tau2_nexus_pert) / A_nexus

    # soma
    GK = gK * np.power(n, 4.0)
    GNa = gNa * np.power(m, 3.0) * h
    GL = gL

    I_I_ext_soma = pf.I_ext_constant(amp_soma) / A_soma

    # dVm_soma/dt
    dy[0] = (I_I_ext_soma - (GK * (Vm_soma - E_K)) - (GNa * (Vm_soma - E_Na)) - (GL * (Vm_soma - E_l)) -
             g_con*(Vm_soma - Vm_nexus)) / Cm_soma

    # dn/dt
    dy[1] = (cf.HH_K_alpha_n(Vm_soma) * (1.0 - n)) - (cf.HH_K_beta_n(Vm_soma) * n)

    # dm/dt
    dy[2] = (cf.HH_Na_alpha_m(Vm_soma) * (1.0 - m)) - (cf.HH_Na_beta_m(Vm_soma) * m)

    # dh/dt
    dy[3] = (cf.HH_Na_alpha_h(Vm_soma) * (1.0 - h)) - (cf.HH_Na_beta_h(Vm_soma) * h)

    # dVm_nexus/dt
    dy[4] = (I_I_ext_nexus + I_I_ext_nexus_syn_gen + I_I_ext_nexus_syn_pert - I_Ca_HVA - I_Im - I_l -
             g_con*(Vm_nexus - Vm_soma)) / Cm_nexus

    # Im dm/dt
    alpha, beta = cf.Im_m_alpha_beta_with_params(Vm_nexus, shift, slope, k)
    dy[5] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


def v_ncline(Vs, param_dict):
    '''
    dv/dt=0:
    ========
    0 = ( I_ext - gCa_HVAbar*mCa*hCa*(V-ECa) - gImbar*mIm*(V-EK) - gL*(V-EL) ) / Cm
    -> gCa_HVAbar*mCa*hCa*(V-ECa) + gImbar*mIm*(V-EK) + gL*(V-EL) = I_ext
    -> gImbar*mIm*(V-EK) = I_ext - gCa_HVAbar*mCa*hCa*(V-ECa) - gL*(V-EL)
    -> mIm = ( I_ext - gCa_HVAbar*mCa*hCa*(V-ECa) - gL*(V-EL) ) / gImbar*(V-EK)
    '''

    I_ext = param_dict['I_ext_per_area']
    gCa_HVAbar = param_dict['gCa_HVAbar']
    gImbar = param_dict['gImbar']
    E_Ca = param_dict['E_Ca']
    E_l = param_dict['E_l']
    E_K = param_dict['E_K']
    gl_pas = param_dict['gl_pas']
    Ca_HVA_h_inf = param_dict['Ca_HVA_h_inf']

    Ca_HVA_m_infs = np.array([cf.Ca_HVA_m_inf(Vs_i) for Vs_i in Vs])

    Im_m = (I_ext - gCa_HVAbar * Ca_HVA_m_infs * Ca_HVA_h_inf * (Vs - E_Ca) - gl_pas * (Vs - E_l)) / \
           (gImbar * (Vs - E_K))

    return Im_m


def Im_m_ncline(Vs, param_dict):
    '''
    dIm_m/dt=0:
    ==========
    0 = (m_inf - Im_m)/m_tau
    -> Im_m = m_inf
    -> Im_m = alpha / (alpha + beta)
    -> Im_m = k * np.exp(slope* fac * (v - shift)) / \
            ( k * np.exp(slope* fac * (v - shift)) + k * np.exp(-slope* fac * (v - shift)))
    '''

    fac = 0.04
    shift = param_dict['shift']
    slope = param_dict['slope']
    k = param_dict['k']

    Im_m = k * np.exp(slope * fac * (Vs - shift)) / \
           (k * np.exp(slope * fac * (Vs - shift)) + k * np.exp(-slope * fac * (Vs - shift)))

    return Im_m
