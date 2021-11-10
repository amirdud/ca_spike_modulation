import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# matplotlib.use('Qt5Agg')
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.integrate import odeint
import mod_dynamics_funs as mdf
import funs


# =========== parameters ===========
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_parameters = pickle.load(file)
file.close()

# Set random seed (for reproducibility)
np.random.seed(1000)

# Start and end time (in milliseconds)
tmin = 0.0
tmax = 250
dt = 0.025
T = np.arange(tmin, tmax, dt)

# general parameters
save_fig = False
to_mS = 1000
shift = -39
slope = 2.3
k = 2.0e-3
y_0 = -42

E_K = -85  # mV
E_Ca = 120  # mV
E_l = -77  # mV
Cm = 1.0  # uF/cm2

A = 1e-5  # cm2 (18 um cell diameter)

# simulation parameters
mul_K = 7.5
mul_gpas = 1

# generate spike
ts_gen = 150
tau1_gen = 0.5
tau2_gen = 5

stim_delay = np.copy(ts_gen)

# perturb spike
tau1_pert = 0.5
tau2_pert = 5

V_init = -75
Im_m_init = 0

# pert_type = 'epsp_and_weak_ipsp'
pert_type = 'strong_ipsp'
# pert_type = 'epsp_delta_t'
# pert_type = 'ACh'
# pert_type = 'contant current'

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
        I_syn = amp_syn * ((tau1*tau2)/(tau1-tau2)) *(np.exp(-(t_curr - ts) / tau1) - np.exp(-(t_curr - ts) / tau2))

        return I_syn


def compute_derivatives(y, t_curr, amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                        amp_syn_pert, ts_pert,tau1_pert, tau2_pert):
    dy = np.zeros((2,))

    v         = y[0]
    Im_m      = y[1]

    Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)  # constant
    Ca_HVA_m_inf = mdf.Ca_HVA_m_inf(v)  # instantaneous
    gCa_HVA = mdf.calc_conductance('HVA_first_order', gCa_HVAbar,
                                   [Ca_HVA_m_inf, Ca_HVA_h_inf])  # mS/cm2

    gIm = mdf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = mdf.calc_current(v, gCa_HVA, E_Ca)  # uA/cm2
    I_Im = mdf.calc_current(v, gIm, E_K)
    I_l = mdf.calc_current(v, gl_pas, E_l)
    I_I_ext = I_ext_constant(amp) / A
    I_I_ext_syn_gen = I_ext_syn(t_curr, amp_syn=amp_syn_gen, ts=ts_gen, tau1=tau1_gen,tau2=tau2_gen) / A
    I_I_ext_syn_pert = I_ext_syn(t_curr, amp_syn=amp_syn_pert, ts=ts_pert, tau1=tau1_pert, tau2=tau2_pert) / A

    # dv/dt
    dy[0] = (I_I_ext + I_I_ext_syn_gen + I_I_ext_syn_pert - I_Ca_HVA - I_Im - I_l) / Cm

    # Im dm/dt
    alpha, beta = mdf.Im_m_alpha_beta_modified(v, shift, slope, k)
    dy[1] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


if pert_type == 'epsp_and_weak_ipsp':
    amp = 0
    mul_Ca = 5.1

    amp_syn_gen = 0.0003  # uA
    ts_pert = 165
    amp_syn_perts_epsp = np.round(np.arange(0, 0.0005+0.00005, 0.00005), 5)  # uA
    amp_syn_perts_weak_ipsp = np.round(np.arange(-0.0003, 0, 0.00005), 5)  # uA
    amp_syn_perts = np.concatenate((amp_syn_perts_weak_ipsp, amp_syn_perts_epsp))
    amp_syn_perts_list = [amp_syn_perts_weak_ipsp, amp_syn_perts_epsp]

    lbox = 31
    rbox = 36
    ubox = -15
    dbox = -90

    xlim = [-35, 55]
    ylim = [15, 35]

    p_vector = np.copy(amp_syn_perts)

    # cmap
    coolwarms = cm.get_cmap('coolwarm', 256)
    newcolors_reds = coolwarms(np.linspace(0.5, 1, 256))  # PSC: 0 - 0.5
    newcolors_weak_blues = coolwarms(np.linspace((0.0009 - 0.0003) / 0.0009 * 0.5, (0.0009 - 0.00005) / 0.0009 * 0.5, 256))  # PSC: -1.75 - -0.25
    weak_blues_part = ListedColormap(newcolors_weak_blues)
    reds_part = ListedColormap(newcolors_reds)
    weak_blues_part_interval = np.linspace(0, 1, len(amp_syn_perts_weak_ipsp))
    reds_part_interval = np.linspace(0, 1, len(amp_syn_perts_epsp))

    colors_weak_blues = [weak_blues_part(k) for k in weak_blues_part_interval]
    colors_reds = [reds_part(k) for k in reds_part_interval]
    colors = colors_weak_blues + colors_reds

elif pert_type == 'strong_ipsp':
    amp = 0
    mul_Ca = 5.1

    amp_syn_gen = 0.0003  # uA
    ts_pert = 165
    amp_syn_perts_strong_ipsp = np.round(np.arange(-0.0009, -0.00076, 0.00002), 5)

    lbox = 15
    rbox = 35
    ubox = -10
    dbox = -95

    xlim = [-90,-76]
    ylim = [13, 15]

    p_vector = np.copy(amp_syn_perts_strong_ipsp)

    # cmap
    coolwarms = cm.get_cmap('coolwarm', 256)
    newcolors_strong_blues = coolwarms(np.linspace(0, (0.0009 - 0.00078) / 0.0009 * 0.5, 256))  # PSC: -1.75 - -0.25
    strong_blues_part = ListedColormap(newcolors_strong_blues)
    strong_blues_part_interval = np.linspace(0, 1, len(amp_syn_perts_strong_ipsp))

    colors = [strong_blues_part(k) for k in strong_blues_part_interval]

elif pert_type == 'epsp_delta_t':
    amp = 0
    mul_Ca = 5.1

    amp_syn_gen = 0.0003  # uA
    ts_perts = np.arange(160, 180, 2)
    amp_syn_pert = 0.0005  # uA

    lbox = 31
    rbox = 38
    ubox = -15
    dbox = -90

    xlim = [-35, 55]

    p_vector = np.copy(ts_perts)

    # cmap
    colormap = cm.BuPu
    evenly_spaced_interval = np.linspace(0, 1, len(p_vector))
    colors = [colormap(k) for k in evenly_spaced_interval]

elif pert_type == 'ACh':
    amp = 0
    mul_Cas = np.arange(5.1, 5.61, 0.05)

    amp_syn_gen = 0.0003
    ts_pert = 165
    amp_syn_pert = 0

    lbox = 31
    rbox = 46
    ubox = -10
    dbox = -90

    xlim = [-35, 55]

    p_vector = np.copy(mul_Cas)

    # cmap
    coolwarm = cm.get_cmap('coolwarm', 256)
    newcolors = coolwarm(np.linspace(0.65, 0.95, 256))
    colormap = ListedColormap(newcolors)

    evenly_spaced_interval = np.linspace(0, 1, len(p_vector))
    colors = [colormap(k) for k in evenly_spaced_interval]


elif pert_type == 'contant current':
    amps = np.round(np.arange(0.00002, 0.00012, 0.00001), 5)
    mul_Ca = 5.1

    amp_syn_gen = 0
    ts_pert = 165
    amp_syn_pert = 0

    lbox = 30
    rbox = 36
    ubox = -15
    dbox = -87

    xlim = [-35, 55]

    p_vector = np.copy(amps)

    # cmap
    coolwarm = cm.get_cmap('coolwarm', 256)
    newcolors = coolwarm(np.linspace(0.65, 0.95, 256))
    colormap = ListedColormap(newcolors)
    evenly_spaced_interval = np.linspace(0, 1, len(p_vector))
    colors = [colormap(k) for k in evenly_spaced_interval]


# ============ run simulation =============
dic_log = {}
for i, p in enumerate(p_vector):
    tic = time.time()
    data = {}

    if pert_type == 'ACh':
        gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * p * to_mS
    else:
        gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS

    gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
    gl_pas = nexus_parameters['g_pas'] * mul_gpas * to_mS

    # State vector parameters: v, all gates
    Y = np.array([V_init, Im_m_init])

    # Solve ODE system
    if pert_type == 'epsp_and_weak_ipsp' or pert_type == 'strong_ipsp':
        Vy = odeint(compute_derivatives, Y, T, args=(amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                                     p, ts_pert, tau1_pert, tau2_pert))
    elif pert_type == 'epsp_delta_t':
        Vy = odeint(compute_derivatives, Y, T, args=(amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                                     amp_syn_pert, p, tau1_pert, tau2_pert))
    elif pert_type == 'contant current':
        Vy = odeint(compute_derivatives, Y, T, args=(p, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                                     amp_syn_pert, ts_pert, tau1_pert, tau2_pert))
    else:
        Vy = odeint(compute_derivatives, Y, T, args=(amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                                     amp_syn_pert, ts_pert, tau1_pert, tau2_pert))

    # get derivatives
    dy_list = []
    for i in range(len(T)):
        if pert_type == 'epsp_and_weak_ipsp' or pert_type == 'strong_ipsp':
            dy = compute_derivatives(Vy[i], T[i], amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                     p, ts_pert, tau1_pert, tau2_pert)
        elif pert_type == 'epsp_delta_t':
            dy = compute_derivatives(Vy[i], T[i], amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                     amp_syn_pert, p, tau1_pert, tau2_pert)
        elif pert_type == 'contant current':
            dy = compute_derivatives(Vy[i], T[i], p, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                     amp_syn_pert, ts_pert, tau1_pert, tau2_pert)
        else:
            dy = compute_derivatives(Vy[i], T[i], amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                                     amp_syn_pert, ts_pert, tau1_pert, tau2_pert)
        dy_list.append(dy)

    Vdy = np.array(dy_list)
    v_dot_t = Vdy[:, 0]
    Im_m_dot_t = Vdy[:, 1]

    v_t = Vy[:, 0]
    Im_m_t = Vy[:, 1]

    Ca_HVA_h_inf_t = np.ones_like(v_t) * mdf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf_t = np.array([mdf.Ca_HVA_m_inf(v_ti) for v_ti in v_t])
    gCa_HVA_t = mdf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf_t, Ca_HVA_h_inf_t])
    gIm_t = mdf.calc_conductance('Im', gImbar, [Im_m_t])
    gl_pas_t = gl_pas * np.ones(np.size(v_t))

    I_Ca_HVA_t = mdf.calc_current(v_t, gCa_HVA_t, E_Ca)
    I_Im_t = mdf.calc_current(v_t, gIm_t, E_K)
    if pert_type == 'Vrest':
        I_l_t = mdf.calc_current(v_t, gl_pas_t, p)
    else:
        I_l_t = mdf.calc_current(v_t, gl_pas_t, E_l)

    # DC stim
    if pert_type == 'contant current':
        I_ext_per_area = I_ext_constant(p) / A
    else:
        I_ext_per_area = I_ext_constant(amp) / A

    # generation stim
    I_ext_syn_gen_vec = np.array([I_ext_syn(t_curr, amp_syn_gen, ts_gen, tau1_gen, tau2_gen)/A for t_curr in T])

    # perturbation stim
    if pert_type == 'epsp_and_weak_ipsp' or pert_type == 'strong_ipsp':
        I_ext_syn_pert_vec = np.array([I_ext_syn(t_curr, p, ts_pert, tau1_pert, tau2_pert)/A for t_curr in T])
    elif pert_type == 'epsp_delta_t':
        I_ext_syn_pert_vec = np.array([I_ext_syn(t_curr, amp_syn_pert, p, tau1_pert, tau2_pert)/A for t_curr in T])
    else:
        I_ext_syn_pert_vec = np.array([I_ext_syn(t_curr, amp_syn_pert, ts_pert, tau1_pert, tau2_pert)/A for t_curr in T])

    I_ext_sum_vec = I_ext_per_area * np.ones_like(I_ext_syn_gen_vec) + I_ext_syn_gen_vec + I_ext_syn_pert_vec

    data['v'] = v_t
    data['t'] = T

    data['I_ext_per_area'] = I_ext_per_area
    data['I_ext_syn_gen_vec'] = I_ext_syn_gen_vec
    data['I_ext_syn_pert_vec'] = I_ext_syn_pert_vec
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

    dic_log[p] = data

    toc = time.time()
    print('finished %f parameter %f secs' % (p, toc - tic))


# ============ get properties ============
# Note: for amps we need to align ca spikes to check duration

props = {}

for p in p_vector:
    props_curr = {}

    v_np = dic_log[p]['v']
    t_np = np.copy(T)

    (log,start_t,stop_t) = funs.is_calcium_spike(t_np,v_np)
    props_curr['log'] = log

    if log:
        ca_spike_dur = stop_t - start_t

        [dummy,start_ind] = funs.find_nearest(t_np,start_t)
        [dummy,stop_ind] = funs.find_nearest(t_np,stop_t)
        area = funs.calc_integral(t_np,v_np,start_ind,stop_ind,y_0)

        # max plateau value in specific t
        t_plat = 36.8 / 2  # middle point in the first plateau in the loop
        [dummy, ind_plat] = funs.find_nearest(t_np[start_ind:]-t_np[start_ind:][0], t_plat) # the ca spike moves so we align it and take a value in the middle of the first spike
        max_plat = v_np[start_ind:][ind_plat]

        # ind of Ca spike peak
        ind_peak = np.argmax(v_np)

        props_curr['area'] = area
        props_curr['duration'] = ca_spike_dur
        props_curr['start_t'] = start_t
        props_curr['stop_t'] = stop_t
        props_curr['max_plat'] = max_plat
        props_curr['ind_peak'] = ind_peak

    else:
        ca_spike_dur = 0

    props[p] = props_curr


#%%
# ========= plot =============
fig, ax = plt.subplots(figsize=(4, 2.5))

for i,p in enumerate(p_vector):
    c = colors[i]
    if pert_type == 'epsp_and_weak_ipsp' or pert_type == 'strong_ipsp' or pert_type == 'ACh' or pert_type == 'epsp_delta_t':
        if p == 0:
            c = 'k'
        plt.plot(dic_log[p]['t'] - stim_delay,dic_log[p]['v'],color=c)

    if pert_type == 'contant current':
        dt = 0.025
        bef_peak_t = 5  # msec
        bef_peak_ind = int(bef_peak_t / dt)

        ind_peak = props[p]['ind_peak']
        t_peak = ind_peak * dt
        plt.plot(t_np - t_peak, dic_log[p]['v'], linewidth=2, color=colors[i], label='v')

plt.plot([lbox, lbox], [dbox, ubox], '--k')
plt.plot([rbox, rbox], [dbox, ubox], '--k')
plt.plot([lbox, rbox], [dbox, dbox], '--k')
plt.plot([lbox, rbox], [ubox, ubox], '--k')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('ms', fontsize=16)
ax.set_ylabel('mV', fontsize=16)
ax.set_xlim([-5, 55])
ax.set_ylim([-100, 50])
ax.tick_params(axis='both', labelsize=12)

if save_fig:
    fig.savefig("./Fig5/fig5_" + pert_type + ".svg", format='svg')


# ============ quantify ===============
list_dur = []
for p in p_vector:
    dur = props[p]['duration']
    list_dur.append(dur)

fig,ax = plt.subplots(figsize=(3.5,1))
for i, p in enumerate(p_vector):
    c = colors[i]
    if pert_type == 'ACh':
        plt.plot(p / 5.1, list_dur[i], 'o', color=c, markersize=8)
    else:
        plt.plot(p, list_dur[i], 'o', color=c, markersize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('ms', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
if pert_type == 'strong_ipsp':
    ax.set_xlim([-0.00091, -0.00078])
    ax.set_ylim([13, 15])

plt.show()
if save_fig:
    fig.savefig("./Fig5/fig5_" + pert_type + "_quantify.svg", format='svg')
