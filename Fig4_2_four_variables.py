import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
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
E_K = -77.0
E_Ca = 132
Cm = 1.0

A = 1e-5  # cm2 (18 um cell diameter)

# simulation parameters
amps = [0]
E_ls = [-75]
mul_Cas = [5]
mul_Ks = [7.5]

ts = 10
tau1 = 0.5
tau2 = 5
amp_syn = 0.0004 # uA
stim_delay = 10

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
    dy = np.zeros((4,))

    v         = y[0]
    Ca_HVA_m  = y[1]
    Ca_HVA_h  = y[2]
    Im_m      = y[3]

    gCa_HVA = mdf.calc_conductance('HVA', gCa_HVAbar, [Ca_HVA_m, Ca_HVA_h])

    gIm = mdf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = mdf.calc_current(v, gCa_HVA, E_Ca)
    I_Im = mdf.calc_current(v, gIm, E_K)
    I_l = mdf.calc_current(v, gl_pas, E_l)
    I_I_ext = I_ext_constant(amp) / A
    I_I_ext_syn = I_ext_syn(t_curr, amp_syn=amp_syn, ts=ts, tau1=tau1,tau2=tau2) / A

    # dv/dt
    dy[0] = (I_I_ext + I_I_ext_syn - I_Ca_HVA - I_Im - I_l) / Cm

    # Ca_HVA dm/dt
    m_inf = mdf.Ca_HVA_m_inf(v)
    m_tau = mdf.Ca_HVA_m_tau(v)
    dy[1] = (m_inf - Ca_HVA_m)/m_tau

    # Ca_HVA dh/dt
    h_inf = mdf.Ca_HVA_h_inf(v)
    h_tau = mdf.Ca_HVA_h_tau(v)
    dy[2] = (h_inf - Ca_HVA_h)/h_tau

    # Im dm/dt
    alpha, beta = mdf.Im_m_alpha_beta(v)
    dy[3] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy

# ============ run simulation =============
dic_log = {}
for amp in amps:
    for E_l in E_ls:
        for mul_Ca in mul_Cas:
            for mul_K in mul_Ks:
                tic = time.time()
                data = {}

                gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
                gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
                gl_pas = nexus_parameters['g_pas'] * to_mS

                # State vector parameters: v, all gates
                V_init = np.copy(E_l)
                Y = np.array([V_init, mdf.Ca_HVA_m_inf(V_init),
                              mdf.Ca_HVA_h_inf(V_init),mdf.Im_m_inf(V_init)])

                # Solve ODE system
                Vy = odeint(compute_derivatives, Y, T, args=(amp, amp_syn, ts, tau1, tau2))

                # get derivatives
                dy_list = []
                for i in range(len(T)):
                    dy = compute_derivatives(Vy[i], T[i], amp, amp_syn, ts, tau1, tau2)
                    dy_list.append(dy)
                Vdy = np.array(dy_list)
                v_dot_t = Vdy[:, 0]
                Ca_HVA_m_dot_t = Vdy[:, 1]
                Ca_HVA_h_dot_t = Vdy[:, 2]
                Im_m_dot_t = Vdy[:, 3]

                v_t = Vy[:, 0]
                Ca_HVA_m_t = Vy[:, 1]
                Ca_HVA_h_t = Vy[:, 2]
                Im_m_t = Vy[:, 3]

                gCa_HVA_t = mdf.calc_conductance('HVA', gCa_HVAbar, [Ca_HVA_m_t, Ca_HVA_h_t ])
                gIm_t = mdf.calc_conductance('Im', gImbar, [Im_m_t])
                gl_pas_t = gl_pas*np.ones(np.size(v_t))

                I_Ca_HVA_t = mdf.calc_current(v_t, gCa_HVA_t, E_Ca)
                I_Im_t = mdf.calc_current(v_t, gIm_t, E_K)
                I_l_t = mdf.calc_current(v_t, gl_pas_t, E_l)

                # Input stimulus
                I_ext_per_area = I_ext_constant(amp) / A
                I_ext_syn_vec = np.array([I_ext_syn(t_curr, amp_syn, ts, tau1, tau2)/A for t_curr in T])
                I_ext_sum_vec = I_ext_per_area * np.ones_like(I_ext_syn_vec) + I_ext_syn_vec

                data['v'] = v_t
                data['t'] = T

                data['I_ext_per_area'] = I_ext_per_area
                data['I_ext_syn_vec'] = I_ext_syn_vec
                data['I_ext_sum_vec'] = I_ext_sum_vec
                data['I_l_t'] = I_l_t
                data['I_Ca_HVA_t'] = I_Ca_HVA_t
                data['I_Im_t'] = I_Im_t

                data['Ca_HVA_m_t'] = Ca_HVA_m_t
                data['Ca_HVA_h_t'] = Ca_HVA_h_t
                data['Im_m_t'] = Im_m_t

                data['gCa_HVA_t'] = gCa_HVA_t
                data['gIm_t'] = gIm_t
                data['gl_pas_t'] = gl_pas_t

                dic_log[(amp,E_l,mul_Ca,mul_K)] = data

                toc = time.time()
                print('finished %f amp %f E_l %f mul_Ca %f mul_K in %f secs' % (amp, E_l, mul_Ca, mul_K, toc - tic))

# ======== plot ===========
from matplotlib import cm

evenly_spaced_interval = np.linspace(0, 1, 7)
colors = [cm.coolwarm(k) for k in evenly_spaced_interval]

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(4, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1, 0.2]})
for amp in amps:
    for E_l in E_ls:
        for mul_Ca in mul_Cas:
            for mul_K in mul_Ks:
                ax0.plot(dic_log[(amp,E_l,mul_Ca,mul_K)]['t']-stim_delay, dic_log[(amp,E_l,mul_Ca,mul_K)]['v'],color='r')
                line_Ca_HVA, = ax1.plot(dic_log[(amp, E_l, mul_Ca, mul_K)]['t']-stim_delay, dic_log[(amp, E_l, mul_Ca, mul_K)]['Ca_HVA_m_t'],label='mCa',color=colors[6])
                ax1.plot(dic_log[(amp, E_l, mul_Ca, mul_K)]['t']-stim_delay, dic_log[(amp, E_l, mul_Ca, mul_K)]['Ca_HVA_h_t'],'--',label='hCa',color=colors[6],linestyle='dashed')
                line_Im, = ax1.plot(dic_log[(amp, E_l, mul_Ca, mul_K)]['t']-stim_delay, dic_log[(amp, E_l, mul_Ca, mul_K)]['Im_m_t'],label='mIm',color=colors[0])
                ax2.plot(dic_log[(amp, E_l, mul_Ca, mul_K)]['t']-stim_delay,dic_log[(amp, E_l, mul_Ca, mul_K)]['I_ext_sum_vec'],color='r')

ax0.set_ylabel('Vm (mV)', fontsize=16)
ax0.set_xlim([-5, 60])
ax0.set_ylim([-90, 50])

ax1.set_ylabel('gate', fontsize=16)
ax1.set_xlim([-5, 60])
ax1.legend(loc='upper right')

ax2.set_ylabel('I', fontsize=16)
ax2.set_xlabel('ms', fontsize=16)
ax2.set_xlim([-5, 60])

plt.show()
# fig.savefig("Fig4/fig4_CaHVA_Im_two_channels_model.svg", format='svg')
