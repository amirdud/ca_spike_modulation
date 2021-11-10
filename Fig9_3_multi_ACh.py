import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
# matplotlib.use('Qt5Agg')
import pickle
from scipy.integrate import odeint
import mod_dynamics_funs as mdf
import funs

# ========== parameters ============
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_parameters = pickle.load(file)
file.close()

# Set random seed (for reproducibility)
np.random.seed(1)

# general parameters
shift = -39
slope = 2.3
k =2.0e-3
to_mS = 1000
y_0 = -75
tmin = 0.0
tmax = 200.0
dt = 0.025
T = np.arange(tmin, tmax, dt)

# parameters from:
# http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
gK = 36
gNa = 120
gL = 0.3
g_con = 0.4

Cm_nexus = 1.0
Cm_soma = 1.0
A_soma = 1e-5  # cm2 (18 um cell diameter)
A_nexus = 1e-5  # cm2 (18 um cell diameter)

E_K = -77
E_Na = 50
E_l = -54.4
E_Ca = 120

amp_nexus = 0
amp_soma = 0

Im_m_init = 0
mul_gpas = 1
mul_K = 7.5

# generate spike
amp_nexus_syn_gen = 0.0004
ts_nexus_gen = 50
tau1_nexus_gen = 0.5
tau2_nexus_gen = 8

# perturb spike
amp_nexus_syn_pert = 0
ts_nexus_pert = 58
tau1_nexus_pert = 0.5
tau2_nexus_pert = 5

V_soma_init = -65.0
V_nexus_init = -65.0

mul_Ca_list = np.round(np.arange(7, 9.1, 0.1), 1)

# ========== Functions ===============
# K
def alpha_n(Vm):
    return (0.01 * (Vm + 55.0)) / (1 - np.exp(-(Vm + 55.0)/10.0))


def beta_n(Vm):
    return (0.125 * np.exp(-(Vm + 65.0)/80.0))


# Na
def alpha_m(Vm):
    return (0.1 * (Vm + 40.0)) / (1 - np.exp(-(Vm + 40.0)/10.0))


def beta_m(Vm):
    return (4 * np.exp(-(Vm + 65.0)/18.0))


def alpha_h(Vm):
    return 0.07 * np.exp(-(Vm + 65.0) / 20.0)


def beta_h(Vm):
    return (1 / (1 + np.exp(-(Vm + 35.0) / 10.0)) )


def n_inf(Vm=-65.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))


def m_inf(Vm=-65.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))


def h_inf(Vm=-65.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))


def I_ext_constant(amp):
    return amp


def I_ext_pulse(t_curr, amp, ts=50, dur=10):
    if ts < t_curr < ts+dur:
        return amp
    else:
        return 0.0


def I_ext_syn(t_curr, amp_syn, ts=0, tau1=0.5, tau2=5):
    # synapse external current
    if t_curr < ts:
        return 0.0
    else:
        # (a)
        # I_syn = amp_syn*np.exp(-(t_curr - ts) / tau)
        # (b)
        # I_syn = amp_syn * ((t_curr - ts) / tau) *np.exp(-(t_curr - ts) / tau)
        # (c)
        I_syn = amp_syn * ((tau1*tau2)/(tau1-tau2)) *( np.exp(-(t_curr - ts) / tau1) - np.exp(-(t_curr - ts) / tau2))

        return I_syn


def compute_derivatives(y, t_curr, amp_soma, amp_nexus,
                        amp_nexus_syn_gen, ts_nexus_gen, tau1_nexus_gen, tau2_nexus_gen,
                        amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert, tau2_nexus_pert):

    dy = np.zeros((6,))

    Vm_soma = y[0]
    n = y[1]
    m = y[2]
    h = y[3]
    Vm_nexus = y[4]
    Im_m = y[5]

    # nexus
    Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf = mdf.Ca_HVA_m_inf(Vm_nexus) # inst
    gCa_HVA = mdf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])
    gIm = mdf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = mdf.calc_current(Vm_nexus, gCa_HVA, E_Ca)
    I_Im = mdf.calc_current(Vm_nexus, gIm, E_K)
    I_l = mdf.calc_current(Vm_nexus, gl_pas, E_l)
    I_I_ext_nexus = I_ext_constant(amp_nexus) / A_nexus
    I_I_ext_nexus_syn_gen = I_ext_syn(t_curr, amp_syn=amp_nexus_syn_gen, ts=ts_nexus_gen,
                                      tau1=tau1_nexus_gen, tau2=tau2_nexus_gen) / A_nexus
    I_I_ext_nexus_syn_pert = I_ext_syn(t_curr, amp_syn=amp_nexus_syn_pert, ts=ts_nexus_pert,
                                       tau1=tau1_nexus_pert, tau2=tau2_nexus_pert) / A_nexus

    # soma
    GK = gK * np.power(n, 4.0)
    GNa = gNa * np.power(m, 3.0) * h
    GL = gL

    I_I_ext_soma = I_ext_constant(amp_soma) / A_soma

    # dVm_soma/dt
    dy[0] = (I_I_ext_soma - (GK * (Vm_soma - E_K)) - (GNa * (Vm_soma - E_Na)) - (GL * (Vm_soma - E_l)) -
             g_con*(Vm_soma - Vm_nexus)) / Cm_soma

    # dn/dt
    dy[1] = (alpha_n(Vm_soma) * (1.0 - n)) - (beta_n(Vm_soma) * n)

    # dm/dt
    dy[2] = (alpha_m(Vm_soma) * (1.0 - m)) - (beta_m(Vm_soma) * m)

    # dh/dt
    dy[3] = (alpha_h(Vm_soma) * (1.0 - h)) - (beta_h(Vm_soma) * h)

    # dVm_nexus/dt
    dy[4] = (I_I_ext_nexus + I_I_ext_nexus_syn_gen + I_I_ext_nexus_syn_pert - I_Ca_HVA - I_Im - I_l -
             g_con*(Vm_nexus - Vm_soma)) / Cm_nexus

    # Im dm/dt
    alpha, beta = mdf.Im_m_alpha_beta_modified(Vm_nexus, shift, slope, k)
    dy[5] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


# ============= run simulation ==============
dic_log = {}

tic = time.time()
for mul_Ca in mul_Ca_list:

    tic = time.time()
    data = {}

    # State vector parameters: v, all gates
    gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
    gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
    gl_pas = nexus_parameters['g_pas'] * mul_gpas * to_mS

    Y = np.array([V_soma_init, n_inf(V_soma_init), m_inf(V_soma_init), h_inf(V_soma_init),
                  V_nexus_init, Im_m_init])

    # Solve ODE system
    Vy = odeint(compute_derivatives, Y, T, args=(amp_soma, amp_nexus,
                                                 amp_nexus_syn_gen, ts_nexus_gen, tau1_nexus_gen, tau2_nexus_gen,
                                                 amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert, tau2_nexus_pert))

    dy_list = []
    for i in range(len(T)):
        dy = compute_derivatives(Vy[i], T[i], amp_soma, amp_nexus,
                                 amp_nexus_syn_gen, ts_nexus_gen, tau1_nexus_gen, tau2_nexus_gen,
                                 amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert, tau2_nexus_pert)
        dy_list.append(dy)

    Vdy = np.array(dy_list)
    v_soma_dot_t = Vdy[:, 0]
    n_dot_t = Vdy[:, 1]
    m_dot_t = Vdy[:, 2]
    h_dot_t = Vdy[:, 3]
    v_nexus_dot_t = Vdy[:, 4]
    Im_m_dot_t = Vdy[:, 5]

    v_soma_t = Vy[:, 0]
    n_t = Vy[:, 1]
    m_t = Vy[:, 2]
    h_t = Vy[:, 3]
    v_nexus_t = Vy[:, 4]
    Im_m_t = Vy[:, 5]

    # nexus
    Ca_HVA_h_inf_t = np.ones_like(v_nexus_t) * mdf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf_t = np.array([mdf.Ca_HVA_m_inf(v_ti) for v_ti in v_nexus_t])  # inst
    gCa_HVA_t = mdf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf_t, Ca_HVA_h_inf_t])
    gIm_t = mdf.calc_conductance('Im', gImbar, [Im_m_t])
    gl_pas_t = gl_pas * np.ones(np.size(v_nexus_t))

    I_Ca_HVA_t = mdf.calc_current(v_nexus_t, gCa_HVA_t, E_Ca)
    I_Im_t = mdf.calc_current(v_nexus_t, gIm_t, E_K)
    I_l_nexus_t = mdf.calc_current(v_nexus_t, gl_pas_t, E_l)

    # soma
    GK_t = gK * np.power(n_t, 4.0)
    GNa_t = gNa * np.power(m_t, 3.0) * h_t
    GL_t = gL * np.ones(np.size(v_soma_t))

    IK_t = mdf.calc_current(v_soma_t, GK_t, E_K)
    INa_t = mdf.calc_current(v_soma_t, GNa_t, E_Na)
    I_l_soma_t = mdf.calc_current(v_soma_t, GL_t, E_l)

    # Input stimulus
    I_ext_nexus_per_area = I_ext_constant(amp_nexus) / A_nexus
    I_ext_nexus_syn_gen_vec = np.array([I_ext_syn(t_curr, amp_nexus_syn_gen, ts_nexus_gen, tau1_nexus_gen,
                                                  tau2_nexus_gen) for t_curr in T])
    I_ext_nexus_syn_pert_vec = np.array([I_ext_syn(t_curr, amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert,
                                                   tau2_nexus_pert) for t_curr in T])
    I_ext_nexus_sum_vec = I_ext_nexus_per_area * np.ones_like(I_ext_nexus_syn_gen_vec) + \
                          I_ext_nexus_syn_gen_vec + I_ext_nexus_syn_pert_vec

    data['v_soma_t'] = v_soma_t
    data['n_t'] = n_t
    data['m_t'] = m_t
    data['h_t'] = h_t
    data['v_nexus_t'] = v_nexus_t
    data['t'] = T

    data['I_l_nexus_t'] = I_l_nexus_t
    data['I_l_soma_t'] = I_l_soma_t
    data['IK_t'] = IK_t
    data['INa_t'] = INa_t

    data['GK_t'] = GK_t
    data['GNa_t'] = GNa_t
    data['GL_t'] = GL_t

    data['I_ext_nexus_per_area'] = I_ext_nexus_per_area
    data['I_ext_nexus_syn_gen_t'] = I_ext_nexus_syn_gen_vec
    data['I_ext_nexus_syn_pert_t'] = I_ext_nexus_syn_pert_vec
    data['I_ext_nexus_sum_vec_t'] = I_ext_nexus_sum_vec

    data['v_soma_dot_t'] = v_soma_dot_t
    data['n_dot_t'] = n_dot_t
    data['m_dot_t'] = m_dot_t
    data['h_dot_t'] = h_dot_t
    data['v_nexus_dot_t'] = v_nexus_dot_t
    data['Im_m_dot_t'] = Im_m_dot_t

    data['gCa_HVAbar'] = gCa_HVAbar

    dic_log[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)] = data

    toc = time.time()
    print(f'finished ({mul_Ca}, {amp_nexus_syn_pert}, {ts_nexus_pert})')


# =========== get spike properties ===========
props = {}

for mul_Ca in mul_Ca_list:
    props_curr = {}

    soma_v_np = dic_log[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['v_soma_t']
    nexus_v_np = dic_log[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['v_nexus_t']

    (log, start_t, stop_t) = funs.is_calcium_spike(T, nexus_v_np)
    props_curr['log'] = log

    if log:
        ca_spike_dur = stop_t - start_t

        [dummy, start_ind] = funs.find_nearest(T, start_t)
        [dummy, stop_ind] = funs.find_nearest(T, stop_t)
        area = funs.calc_integral(T, nexus_v_np, start_ind, stop_ind, y_0)
        n_sp = funs.n_spikes(soma_v_np)

        props_curr['area'] = area
        props_curr['duration'] = ca_spike_dur
        props_curr['n_sp'] = n_sp
        props_curr['start_t'] = start_t
        props_curr['stop_t'] = stop_t

    else:
        ca_spike_dur = 0

    props[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)] = props_curr

# ========== plot ===========
gCa_multiplier_list = []
gCa_HVAbar_list = []
list_n_sp = []
for mul_Ca in mul_Ca_list:
    n_sp = props[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['n_sp']
    gCa_HVAbar = dic_log[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['gCa_HVAbar']
    list_n_sp.append(n_sp)
    gCa_HVAbar_list.append(gCa_HVAbar)

gCa_HVAbar_list = np.array(gCa_HVAbar_list)
gCa_multiplier_list = gCa_HVAbar_list/gCa_HVAbar_list[0]

fig, ax = plt.subplots(figsize=(1, 2))
plt.plot(gCa_multiplier_list, list_n_sp, 'ok', markersize=2)
plt.show()
# fig.savefig("./Fig9/Fig9_ACh_summary.svg", format='svg')
