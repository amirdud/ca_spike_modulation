import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
# matplotlib.use('Qt5Agg')

from scipy.integrate import odeint
import mod_dynamics_funs as mdf

# =========== parameters ===========
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_parameters = pickle.load(file)
file.close()

# Set random seed (for reproducibility)
np.random.seed(1000)

# Start and end time (in milliseconds)
tmin = 0.0
tmax = 150
dt = 0.025
T = np.arange(tmin, tmax, dt)

# general parameters
to_mS = 1000
E_K = -85.0
E_Ca = 120
Cm = 1.0
stim_delay = 10
shift = -39
slope = 2.3
k = 2.0e-3

A = 1e-5  # cm2 (18 um cell diameter)

# simulation parameters
amp = 0
E_l = -77
mul_Ca = 5.1
mul_K = 7.5
mul_gpas = 1

ts = 10
tau1 = 0.5
tau2 = 5
amp_syn = 0.0003
V_init = -75
Im_m_init = 0

# =========== functions ===========
def I_ext_constant(amp):
    return amp


def I_ext_pulse(t_curr, amp, ts=0, dur=0):
    if ts < t_curr < ts + dur:
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


def compute_derivatives(y, t_curr, amp, amp_syn, ts, tau1, tau2):
    dy = np.zeros((2,))

    v         = y[0]
    Im_m      = y[1]

    Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf = mdf.Ca_HVA_m_inf(v) # inst
    gCa_HVA = mdf.calc_conductance('HVA_first_order',gCa_HVAbar,[Ca_HVA_m_inf , Ca_HVA_h_inf])

    gIm = mdf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = mdf.calc_current(v,gCa_HVA,E_Ca)
    I_Im = mdf.calc_current(v, gIm, E_K)
    I_l = mdf.calc_current(v, gl_pas, E_l)
    I_I_ext = I_ext_constant(amp) / A
    I_I_ext_syn = I_ext_syn(t_curr, amp_syn=amp_syn, ts=ts, tau1=tau1,tau2=tau2) / A

    # dv/dt
    dy[0] = (I_I_ext + I_I_ext_syn - I_Ca_HVA - I_Im - I_l) / Cm

    # Im dm/dt
    alpha, beta = mdf.Im_m_alpha_beta_modified(v, shift, slope, k)
    dy[1] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy

# ============ run simulation =============
dic_log = {}

tic = time.time()
data = {}

gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
gl_pas = nexus_parameters['g_pas'] * mul_gpas * to_mS

# State vector parameters: v, all gates
Y = np.array([V_init, Im_m_init])

# Solve ODE system
Vy = odeint(compute_derivatives, Y, T, args=(amp, amp_syn, ts, tau1, tau2))

# get derivatives
dy_list = []
for i in range(len(T)):
    dy = compute_derivatives(Vy[i], T[i], amp, amp_syn, ts, tau1, tau2)
    dy_list.append(dy)
Vdy = np.array(dy_list)
v_dot_t = Vdy[:, 0]
Im_m_dot_t = Vdy[:, 1]

v_t = Vy[:, 0]
Im_m_t = Vy[:, 1]

Ca_HVA_h_inf_t = np.ones_like(v_t) * mdf.Ca_HVA_h_inf(E_l)
Ca_HVA_m_inf_t = np.array([mdf.Ca_HVA_m_inf(v_ti) for v_ti in v_t])  # inst
gCa_HVA_t = mdf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf_t, Ca_HVA_h_inf_t])
gIm_t = mdf.calc_conductance('Im', gImbar, [Im_m_t])
gl_pas_t = gl_pas * np.ones(np.size(v_t))

I_Ca_HVA_t = mdf.calc_current(v_t, gCa_HVA_t, E_Ca)
I_Im_t = mdf.calc_current(v_t, gIm_t, E_K)
I_l_t = mdf.calc_current(v_t, gl_pas_t, E_l)

# Input stimulus
I_ext_per_area = I_ext_constant(amp)/A
I_ext_syn_vec = np.array([I_ext_syn(t_curr, amp_syn, ts, tau1, tau2) for t_curr in T])
I_ext_sum_vec = I_ext_per_area * np.ones_like(I_ext_syn_vec) + I_ext_syn_vec

data['v'] = v_t
data['t'] = T

data['I_ext_per_area'] = I_ext_per_area
data['I_ext_syn_vec'] = I_ext_syn_vec
data['I_ext_sum_vec'] = I_ext_sum_vec
data['I_l_t'] = I_l_t
data['I_Ca_HVA_t'] = I_Ca_HVA_t
data['I_Im_t'] = I_Im_t

data['Ca_HVA_m_inf_t'] = Ca_HVA_m_inf_t
data['Ca_HVA_h_inf_t'] = Ca_HVA_h_inf_t
data['Im_m_t'] = Im_m_t

data['gCa_HVA_t'] = gCa_HVA_t
data['gIm_t'] = gIm_t
data['gl_pas_t'] = gl_pas_t

data['v_dot_t'] = v_dot_t
data['Im_m_dot_t'] = Im_m_dot_t

toc = time.time()
print('finished %f amp %f E_l %f mul_Ca %f mul_K in %f secs' % (amp, E_l, mul_Ca, mul_K, toc - tic))

# ======== plot ========

evenly_spaced_interval = np.linspace(0, 1, 7)
colors = [cm.coolwarm(k) for k in evenly_spaced_interval]

fig, ax0 = plt.subplots(figsize=(3, 2))
ax0.plot(data['t'] - stim_delay, data['gCa_HVA_t'], label='gCaHVA', color=colors[6])
ax0.plot(data['t'] - stim_delay, data['gIm_t'], label='gIm', color=colors[0])
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.set_xlim([-10, 70])
ax0.set_ylabel('conductance')
# ax0.legend()
plt.show()
# fig.savefig("./FigS2/figS2_reduced_conductances.svg", format='svg')

fig, ax1 = plt.subplots(figsize=(3, 2))
ax1.plot(data['t'] - stim_delay, data['v'], 'k')
ax1.set_ylabel('mV', fontsize=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_ylim([-90, 50])
ax1.set_xlim([-10, 70])
plt.show()
# fig.savefig("./FigS2/figS2_reduced_voltages.svg", format='svg')

fig, ax2 = plt.subplots(figsize=(3, 0.5))
ax2.plot(data['t'] - stim_delay, data['I_ext_sum_vec'], color='k')
ax2.set_ylabel('uA', fontsize=16)
ax2.set_xlabel('ms', fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xlim([-10, 70])
plt.show()
# fig.savefig("./FigS2/figS2_reduced_input.svg", format='svg')

fig, ax3 = plt.subplots(figsize=(3, 2))
ax3.plot(data['t'] - stim_delay, data['I_Ca_HVA_t']*(-1), label='I_CaHVA', color=colors[6])
ax3.plot(data['t'] - stim_delay, data['I_Im_t']+data['I_l_t'], label='I_Im', color=colors[0])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.set_xlim([-10, 70])
ax3.set_ylabel('current')
plt.show()
# fig.savefig("./FigS2/figS2_reduced_currents.svg", format='svg')

fig, ax4 = plt.subplots(figsize=(3, 2))
ax4.plot(data['t'] - stim_delay, data['I_Ca_HVA_t'] + data['I_Im_t']+data['I_l_t'])
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.set_xlim([-10, 70])
ax4.set_ylim([-10, 10])
ax4.set_ylabel('sum currents')
plt.show()

fig, ax5 = plt.subplots(figsize=(3, 2))
ax5.plot(data['t'] - stim_delay, (data['v']-E_Ca)*(-1), label='CaHVA', color=colors[6])
ax5.plot(data['t'] - stim_delay, (data['v']-E_K) + (data['v']-E_l), label='Im + Leak', color=colors[0])
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.set_xlim([-10, 70])
ax5.set_ylabel('driving force')
plt.show()
