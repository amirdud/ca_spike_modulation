import time
import pickle
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
# matplotlib.use('Qt5Agg')

from scipy.integrate import odeint
import mod_dynamics_funs as mdf
import funs

# ================ parameters =================
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_parameters = pickle.load(file)
file.close()

# Set random seed (for reproducibility)
np.random.seed(1000)

# Start and end time (in milliseconds)
tmin = 0.0
tmax = 300.0
dt = 0.0025
T = np.arange(tmin, tmax, dt)

to_mS = 1000
shift = -39
slope = 2.3
k =2.0e-3

E_K = -85  # mV
E_Ca = 120  # mV
E_l = -77  # mV
Cm = 1.0  # uF/cm2

A = 1e-5  # cm2 (18 um cell diameter)

amp = 0
mul_K = 7.5
mul_gpas = 1

mul_Cas = np.array([5.1, 6.1, 7.1])

V_init = -30
Im_m_init = 0


# ============== functions =============
def I_ext(amp):
    return amp


def compute_derivatives(y, t0, amp):
    dy = np.zeros((2,))

    v         = y[0]
    Im_m      = y[1]

    Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf = mdf.Ca_HVA_m_inf(v) # inst
    gCa_HVA = mdf.calc_conductance('HVA_first_order',gCa_HVAbar,[Ca_HVA_m_inf , Ca_HVA_h_inf])

    gIm = mdf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = mdf.calc_current(v, gCa_HVA, E_Ca)
    I_Im = mdf.calc_current(v, gIm, E_K)
    I_l = mdf.calc_current(v, gl_pas, E_l)
    I_I_ext = I_ext(amp) / A

    # dv/dt
    dy[0] = (I_I_ext - I_Ca_HVA - I_Im - I_l) / Cm

    # Im dm/dt
    alpha, beta = mdf.Im_m_alpha_beta_modified(v, shift, slope, k)
    dy[1] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy

# =============== simulation ============
dic_log = {}
for mul_Ca in mul_Cas:
    tic = time.time()
    data = {}

    gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
    gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
    gl_pas = nexus_parameters['g_pas'] * mul_gpas * to_mS

    # State vector parameters: v, all gates
    Y = np.array([V_init, Im_m_init])

    # Solve ODE system
    Vy = odeint(compute_derivatives, Y, T, args=(amp,))

    # get derivatives
    dy_list = []
    for i in range(len(T)):
        dy = compute_derivatives(Vy[i], T[i], amp)
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

    I_ext_per_area = I_ext(amp)/A

    data['v'] = v_t
    data['t'] = T

    data['I_ext_per_area'] = I_ext_per_area
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

    dic_log[mul_Ca] = data

    toc = time.time()
    print('finished %f amp %f E_l %f mul_Ca %f mul_K in %f secs' % (amp, E_l, mul_Ca, mul_K, toc - tic))

# ========= plot ==========
from matplotlib import cm

reds = cm.get_cmap('Reds', 256)
greens = cm.get_cmap('Greens', 256)
newcolors_reds = reds(np.linspace(0.3, 0.9, 256))
newcolors_greens = greens(np.linspace(0.3, 0.9, 256))
reds_half = ListedColormap(newcolors_reds)
greens_half = ListedColormap(newcolors_greens)
evenly_spaced_interval = np.linspace(0, 1, len(mul_Cas))
colors_reds = [reds_half(k) for k in evenly_spaced_interval]
colors_greens = [greens_half(k) for k in evenly_spaced_interval]

fig, (ax0) = plt.subplots(1, 1, figsize=(4, 2))
for j, mul_Ca in enumerate(mul_Cas):
    ax0.plot(dic_log[mul_Ca]['t'], dic_log[mul_Ca]['v'], color=colors_greens[j])

ax0.set_xlabel('ms',fontsize = 16)
ax0.set_ylabel('Vm (mV)', fontsize=16)
# ax0.set_xlim([0,80])
ax0.set_ylim([-100, 50])
plt.show()
# fig.savefig("./Fig8/fig8_add_ACh_spike_R2.svg", format='svg')


# ========== nullclines ===========
Im_ms_low = 0
Im_ms_high = 1
Im_ms_delta = 0.0001
Vs_low = -150
Vs_high = 100
Vs_delta = 0.025
n_samples = 35
scale = 10000
dd = 1
ss = 2
lw = 2

Im_ms = np.arange(Im_ms_low, Im_ms_high+Im_ms_delta, Im_ms_delta )
Vs = np.arange(Vs_low, Vs_high+Vs_delta, Vs_delta)
Vs_tot_vals = Vs.shape[0]
Im_ms_tot_vals = Im_ms.shape[0]

Vs_dsamp = int(np.ceil(Vs_tot_vals / n_samples))
Im_ms_dsamp = int(np.ceil(Im_ms_tot_vals / n_samples))

Vs_samp_inds = np.arange(1, Vs_tot_vals, Vs_dsamp)
Im_ms_samp_inds = np.arange(1, Im_ms_tot_vals, Im_ms_dsamp)

Vs_samp = Vs[Vs_samp_inds]
Im_ms_samp = Im_ms[Im_ms_samp_inds]

samp_matrix = [Vs_samp.T, Im_ms_samp.T]
x = np.array([p for p in itertools.product(*samp_matrix)])

Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
gl_pas = nexus_parameters['g_pas'] * mul_gpas * to_mS
I_ext_per_area = I_ext(amp)/A
v_ncline_param_dict = {
    'I_ext_per_area': I_ext_per_area,
    'gImbar': gImbar,
    'E_Ca': E_Ca,
    'E_l': E_l,
    'E_K': E_K,
    'gl_pas': gl_pas,
    'Ca_HVA_h_inf': Ca_HVA_h_inf,
}

Im_m_ncline_param_dict = {
    'shift': shift,
    'slope': slope,
    'k': k,
}
ind_singular = int(np.where(abs(Vs - E_K) < 0.001)[0])

def calc_x_dot(x):
    x_dot = np.zeros_like(x)

    n_conditions = x.shape[0]

    for i in range(n_conditions):
        v = x[i, 0]
        Im_m = x[i, 1]

        Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
        Ca_HVA_m_inf = mdf.Ca_HVA_m_inf(v)

        gCa_HVA = mdf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])

        gIm = mdf.calc_conductance('Im', gImbar, [Im_m])

        I_Ca_HVA = mdf.calc_current(v, gCa_HVA, E_Ca)
        I_Im = mdf.calc_current(v, gIm, E_K)
        I_l = mdf.calc_current(v, gl_pas, E_l)
        I_I_ext = I_ext(amp) / A

        # dv/dt
        v_dot = (I_I_ext - I_Ca_HVA - I_Im - I_l)/Cm

        # Im dm/dt
        alpha, beta = mdf.Im_m_alpha_beta_modified(v,shift,slope,k)
        Im_m_dot = (alpha * (1.0 - Im_m)) - (beta * Im_m)

        x_dot[i, :] = np.hstack((v_dot, Im_m_dot))

    return x_dot


# draw nullclines
# ===============
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for j, mul_Ca in enumerate(mul_Cas):
    gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
    v_ncline_param_dict['gCa_HVAbar'] = gCa_HVAbar

    yIm_m_vnull = funs.v_ncline(Vs, v_ncline_param_dict)
    yIm_m_mnull = funs.Im_m_ncline(Vs, Im_m_ncline_param_dict)

    # draw nullclines
    ax.plot(Vs[ind_singular + 1:], yIm_m_vnull[ind_singular + 1:],
            color=colors_reds[j], label='v-null', linewidth=lw)
    ax.plot(Vs, yIm_m_mnull,
            color=[0.4, 0.4, 0.4], linewidth=lw, label='Im_m-null')
    ax.set_xlabel('mV', fontsize=16)
    ax.set_ylabel('Im_m', fontsize=16)

    # calc field arrows
    x_dot = calc_x_dot(x)

    # draw calcium spike trajectory
    v_t = dic_log[mul_Ca]['v']
    Im_m_t = dic_log[mul_Ca]['Im_m_t']
    xx = np.copy(v_t)[::ss]
    yy = np.copy(Im_m_t)[::ss]
    ax.plot(xx, yy, color=colors_greens[j], markersize=1, linewidth=lw, label='Ca Spike')

    ax.set_xlim([-120, 50])
    ax.set_ylim([0 - 0.01, 1 + 0.01])

plt.show()
# fig.savefig("./Fig8/fig8_phase_plane_add_ACh_R3.svg", format='svg')

