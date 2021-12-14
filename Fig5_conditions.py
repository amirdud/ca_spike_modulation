import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.integrate import odeint
import util_funs as uf
import analysis_funs as af
import phys_funs as pf
import channel_funs as cf
import dynamics_funs as df
from cfg import *
from constants import *

# =========== parameters ===========
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_params = pickle.load(file)
file.close()

general_params = cfg_fig_5_1["h"]
tmin = general_params["tmin"]
tmax = general_params["tmax"]
dt = general_params["dt"]
pert_type = general_params["pert_type"]
t_plat = general_params["t_plat"]
T = np.arange(tmin, tmax, dt)

phys_params = cfg_fig_5_1["phys"]
E_K = phys_params["E_K"]
E_Ca = phys_params["E_Ca"]
E_l = phys_params["E_l"]
Cm = phys_params["Cm"]
A = phys_params["A"]
shift = phys_params["shift"]
slope = phys_params["slope"]
k = phys_params["k"]

sim_params = cfg_fig_5_1["sim"]
mul_K = sim_params["mul_K"]
mul_gpas = sim_params["mul_gpas"]
ts_gen = sim_params["ts_gen"]
tau1_gen = sim_params["tau1_gen"]
tau2_gen = sim_params["tau2_gen"]
stim_delay = sim_params["stim_delay"]
tau1_pert = sim_params["tau1_pert"]
tau2_pert = sim_params["tau2_pert"]
V_init = sim_params["V_init"]
Im_m_init = sim_params["Im_m_init"]

amp = sim_params["pert_dependent_params"][pert_type]["amp"]
mul_Ca = sim_params["pert_dependent_params"][pert_type]["mul_Ca"]
amp_syn_gen = sim_params["pert_dependent_params"][pert_type]["amp_syn_gen"]
ts_pert = sim_params["pert_dependent_params"][pert_type]["ts_pert"]

if pert_type == 'epsp_and_weak_ipsp':
    amp_syn_perts_weak_ipsp = sim_params["pert_dependent_params"][pert_type]["amp_syn_perts_weak_ipsp"]
    amp_syn_perts_epsp = sim_params["pert_dependent_params"][pert_type]["amp_syn_perts_epsp"]

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
    amp_syn_perts_strong_ipsp = sim_params["pert_dependent_params"][pert_type]["amp_syn_perts_strong_ipsp"]

    lbox = 15
    rbox = 35
    ubox = -10
    dbox = -95

    xlim = [-90, -76]
    ylim = [13, 15]

    p_vector = np.copy(amp_syn_perts_strong_ipsp)

    # cmap
    coolwarms = cm.get_cmap('coolwarm', 256)
    newcolors_strong_blues = coolwarms(np.linspace(0, (0.0009 - 0.00078) / 0.0009 * 0.5, 256))  # PSC: -1.75 - -0.25
    strong_blues_part = ListedColormap(newcolors_strong_blues)
    strong_blues_part_interval = np.linspace(0, 1, len(amp_syn_perts_strong_ipsp))

    colors = [strong_blues_part(k) for k in strong_blues_part_interval]

elif pert_type == 'epsp_delta_t':
    amp_syn_pert = sim_params["pert_dependent_params"][pert_type]["amp_syn_pert"]

    lbox = 31
    rbox = 38
    ubox = -15
    dbox = -90

    xlim = [-35, 55]

    p_vector = np.copy(ts_pert)

    # cmap
    colormap = cm.BuPu
    evenly_spaced_interval = np.linspace(0, 1, len(p_vector))
    colors = [colormap(k) for k in evenly_spaced_interval]

elif pert_type == 'ACh':
    amp_syn_pert = sim_params["pert_dependent_params"][pert_type]["amp_syn_pert"]

    lbox = 31
    rbox = 46
    ubox = -10
    dbox = -90

    xlim = [-35, 55]

    p_vector = np.copy(mul_Ca)

    # cmap
    coolwarm = cm.get_cmap('coolwarm', 256)
    newcolors = coolwarm(np.linspace(0.65, 0.95, 256))
    colormap = ListedColormap(newcolors)

    evenly_spaced_interval = np.linspace(0, 1, len(p_vector))
    colors = [colormap(k) for k in evenly_spaced_interval]


elif pert_type == 'constant_current':
    amp_syn_pert = sim_params["pert_dependent_params"][pert_type]["amp_syn_pert"]

    lbox = 30
    rbox = 36
    ubox = -15
    dbox = -87

    xlim = [-35, 55]

    p_vector = np.copy(amp)

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
        gCa_HVAbar = nexus_params['gCa_HVAbar_Ca_HVA'] * p * TO_MS
    else:
        gCa_HVAbar = nexus_params['gCa_HVAbar_Ca_HVA'] * mul_Ca * TO_MS

    gImbar = nexus_params['gImbar_Im'] * mul_K * TO_MS
    gl_pas = nexus_params['g_pas'] * mul_gpas * TO_MS

    # State vector parameters: v, all gates
    Y = np.array([V_init, Im_m_init])

    # Solve ODE system
    if pert_type == 'epsp_and_weak_ipsp' or pert_type == 'strong_ipsp':
        Vy = odeint(df.compute_derivatives_two_variables_model_with_perturbation, Y, T,
                    args=(amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                          p, ts_pert, tau1_pert, tau2_pert,
                          gCa_HVAbar, gImbar, gl_pas,
                          E_Ca, E_K, E_l, A, Cm, shift, slope, k))
    elif pert_type == 'epsp_delta_t':
        Vy = odeint(df.compute_derivatives_two_variables_model_with_perturbation, Y, T,
                    args=(amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                          amp_syn_pert, p, tau1_pert, tau2_pert,
                          gCa_HVAbar, gImbar, gl_pas,
                          E_Ca, E_K, E_l, A, Cm, shift, slope, k))
    elif pert_type == 'constant_current':
        Vy = odeint(df.compute_derivatives_two_variables_model_with_perturbation, Y, T,
                    args=(p, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                          amp_syn_pert, ts_pert, tau1_pert, tau2_pert,
                          gCa_HVAbar, gImbar, gl_pas,
                          E_Ca, E_K, E_l, A, Cm, shift, slope, k))
    else:
        Vy = odeint(df.compute_derivatives_two_variables_model_with_perturbation, Y, T,
                    args=(amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                          amp_syn_pert, ts_pert, tau1_pert, tau2_pert,
                          gCa_HVAbar, gImbar, gl_pas,
                          E_Ca, E_K, E_l, A, Cm, shift, slope, k))
    # get derivatives
    dy_list = []
    for i in range(len(T)):
        if pert_type == 'epsp_and_weak_ipsp' or pert_type == 'strong_ipsp':
            dy = df.compute_derivatives_two_variables_model_with_perturbation(
                Vy[i], T[i], amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                p, ts_pert, tau1_pert, tau2_pert,
                gCa_HVAbar, gImbar, gl_pas,
                E_Ca, E_K, E_l, A, Cm, shift, slope, k)
        elif pert_type == 'epsp_delta_t':
            dy = df.compute_derivatives_two_variables_model_with_perturbation(
                Vy[i], T[i], amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                amp_syn_pert, p, tau1_pert, tau2_pert,
                gCa_HVAbar, gImbar, gl_pas,
                E_Ca, E_K, E_l, A, Cm, shift, slope, k)
        elif pert_type == 'constant_current':
            dy = df.compute_derivatives_two_variables_model_with_perturbation(
                Vy[i], T[i], p, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                amp_syn_pert, ts_pert, tau1_pert, tau2_pert,
                gCa_HVAbar, gImbar, gl_pas,
                E_Ca, E_K, E_l, A, Cm, shift, slope, k)
        else:
            dy = df.compute_derivatives_two_variables_model_with_perturbation(
                Vy[i], T[i], amp, amp_syn_gen, ts_gen, tau1_gen, tau2_gen,
                amp_syn_pert, ts_pert, tau1_pert, tau2_pert,
                gCa_HVAbar, gImbar, gl_pas,
                E_Ca, E_K, E_l, A, Cm, shift, slope, k)
        dy_list.append(dy)

    Vdy = np.array(dy_list)
    v_dot_t = Vdy[:, 0]
    Im_m_dot_t = Vdy[:, 1]

    v_t = Vy[:, 0]
    Im_m_t = Vy[:, 1]

    Ca_HVA_h_inf_t = np.ones_like(v_t) * cf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf_t = np.array([cf.Ca_HVA_m_inf(v_ti) for v_ti in v_t])
    gCa_HVA_t = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf_t, Ca_HVA_h_inf_t])
    gIm_t = pf.calc_conductance('Im', gImbar, [Im_m_t])
    gl_pas_t = gl_pas * np.ones(np.size(v_t))

    I_Ca_HVA_t = pf.calc_current(v_t, gCa_HVA_t, E_Ca)
    I_Im_t = pf.calc_current(v_t, gIm_t, E_K)
    I_l_t = pf.calc_current(v_t, gl_pas_t, E_l)

    # DC stim
    if pert_type == 'constant_current':
        I_ext_per_area = pf.I_ext_constant(p) / A
    else:
        I_ext_per_area = pf.I_ext_constant(amp) / A

    # generation stim
    I_ext_syn_gen_vec = np.array([pf.I_ext_syn(t_curr, amp_syn_gen, ts_gen, tau1_gen, tau2_gen) / A for t_curr in T])

    # perturbation stim
    if pert_type == 'epsp_and_weak_ipsp' or pert_type == 'strong_ipsp':
        I_ext_syn_pert_vec = np.array([pf.I_ext_syn(t_curr, p, ts_pert, tau1_pert, tau2_pert) / A for t_curr in T])
    elif pert_type == 'epsp_delta_t':
        I_ext_syn_pert_vec = np.array([pf.I_ext_syn(t_curr, amp_syn_pert, p, tau1_pert, tau2_pert) / A for t_curr in T])
    else:
        I_ext_syn_pert_vec = np.array([pf.I_ext_syn(t_curr, amp_syn_pert, ts_pert, tau1_pert, tau2_pert)/A for t_curr in T])

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

    (log,start_t,stop_t) = af.is_calcium_spike(t_np, v_np)
    props_curr['log'] = log

    if log:
        ca_spike_dur = stop_t - start_t

        [dummy,start_ind] = uf.find_nearest(t_np, start_t)
        [dummy,stop_ind] = uf.find_nearest(t_np, stop_t)

        # max plateau value in specific t
        [dummy, ind_plat] = uf.find_nearest(t_np[start_ind:] - t_np[start_ind:][0], t_plat) # the ca spike moves so we align it and take a value in the middle of the first spike
        max_plat = v_np[start_ind:][ind_plat]

        # ind of Ca spike peak
        ind_peak = np.argmax(v_np)

        props_curr['duration'] = ca_spike_dur
        props_curr['start_t'] = start_t
        props_curr['stop_t'] = stop_t
        props_curr['max_plat'] = max_plat
        props_curr['ind_peak'] = ind_peak

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

    if pert_type == 'constant_current':
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
plt.show()
if general_params["save_figs"]:
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
if general_params["save_figs"]:
    fig.savefig("./Fig5/fig5_" + pert_type + "_quantify.svg", format='svg')
