import time
import pickle
import itertools
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import util_funs as uf
import phys_funs as pf
import channel_funs as cf
import dynamics_funs as df
from cfg import *
from constants import *

# =========== parameters ===========
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_params = pickle.load(file)
file.close()

general_params = cfg_fig_7_1["h"]
tmin_pre = general_params["tmin_pre"]
tmax_post = general_params["tmax_post"]
dt = general_params["dt"]
t_plat = general_params["t_plat"]
T_full = np.arange(tmin_pre, tmax_post, dt)

phys_params = cfg_fig_7_1["phys"]
E_K = phys_params["E_K"]
E_Ca = phys_params["E_Ca"]
E_l = phys_params["E_l"]
Cm = phys_params["Cm"]
A = phys_params["A"]
shift = phys_params["shift"]
slope = phys_params["slope"]
k = phys_params["k"]
th_end_nexus_mV = phys_params["th_end_nexus_mV"]

sim_params = cfg_fig_7_1["sim"]
amp = sim_params["amp"]
mul_Ca = sim_params["mul_Ca"]
mul_K = sim_params["mul_K"]
mul_gpas = sim_params["mul_gpas"]

# ============ run simulation ===============
# parameters order: [amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition]
condition_pairs = [[amp, E_l, mul_Ca, mul_K, -30, 0, 10, 0],  # ctrl
                   [amp, E_l, mul_Ca, mul_K, -30, 0.1, 10, 0],
                   [amp, E_l, mul_Ca, mul_K, -30, 0.2, 10, 0],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 5, -45],  # termination
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 10, -35],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 2, -65],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 15, -17],  # con. stab. in
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 5, 15],  # con. stab. exc
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 10, 8],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 15, 5],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 5, -15],  # regeneration
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 10, -9],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 15, -5],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 5, -100],  # Reset
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 15, -100]]


dic_log = {}
for pair in condition_pairs:

    amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition = pair

    tic = time.time()
    data = {}

    gCa_HVAbar = nexus_params['gCa_HVAbar_Ca_HVA'] * mul_Ca * TO_MS
    gImbar = nexus_params['gImbar_Im'] * mul_K * TO_MS
    gl_pas = nexus_params['g_pas'] * mul_gpas * TO_MS

    # State vector parameters: v, all gates
    Y_pre = np.array([V_init, Im_m_init])

    # run full
    Vy_full = odeint(df.compute_derivatives_two_variables_model_constant_generation, Y_pre, T_full,
                     args=(amp, gCa_HVAbar, gImbar, gl_pas, E_Ca, E_K, E_l, A, Cm, shift, slope, k))

    v_t_full = Vy_full[:, 0]
    Im_m_t_full = Vy_full[:, 1]
    Ca_HVA_m_inf_t_full = np.array([cf.Ca_HVA_m_inf(v_ti) for v_ti in v_t_full])
    Ca_HVA_h_inf_t_full = np.ones_like(v_t_full) * cf.Ca_HVA_h_inf(E_l)

    # run perturbation
    # Solve ODE system pre perturbation
    tmax_pre = np.copy(timing_perturb)  # 1st parameter
    t_pre = np.arange(tmin_pre, tmax_pre, dt)
    Vy_pre = odeint(df.compute_derivatives_two_variables_model_constant_generation, Y_pre, t_pre,
                    args=(amp, gCa_HVAbar, gImbar, gl_pas, E_Ca, E_K, E_l, A, Cm, shift, slope, k))

    v_t_pre = Vy_pre[:, 0]
    Im_m_t_pre = Vy_pre[:, 1]

    # get derivatives pre
    dy_list_pre = []
    for i in range(len(t_pre)):
        dy_pre = df.compute_derivatives_two_variables_model_constant_generation(
            Vy_pre[i], t_pre[i], amp, gCa_HVAbar, gImbar, gl_pas, E_Ca, E_K, E_l, A, Cm, shift, slope, k)
        dy_list_pre.append(dy_pre)
    Vdy_pre = np.array(dy_list_pre)
    v_dot_t_pre = Vdy_pre[:, 0]
    Im_m_dot_t_pre = Vdy_pre[:, 1]

    # Solve ODE system post perturbation
    Y_post = np.array([v_t_pre[-1] + v_perturb_addition,
                       Im_m_t_pre[-1]])  # 2nd parameter

    tmin_post = np.copy(tmax_pre)
    t_post = np.arange(tmin_post, tmax_post, dt)
    Vy_post = odeint(df.compute_derivatives_two_variables_model_constant_generation, Y_post, t_post,
                     args=(amp, gCa_HVAbar, gImbar, gl_pas, E_Ca, E_K, E_l, A, Cm, shift, slope, k))

    v_t_post = Vy_post[:, 0]
    Im_m_t_post = Vy_post[:, 1]

    # get derivatives post
    dy_list_post = []
    for i in range(len(t_post)):
        dy_post = df.compute_derivatives_two_variables_model_constant_generation(
            Vy_post[i], t_post[i], amp, gCa_HVAbar, gImbar, gl_pas, E_Ca, E_K, E_l, A, Cm, shift, slope, k)
        dy_list_post.append(dy_post)
    Vdy_post = np.array(dy_list_post)
    v_dot_t_post = Vdy_post[:, 0]
    Im_m_dot_t_post = Vdy_post[:, 1]

    v_dot_t = np.hstack((v_dot_t_pre, v_dot_t_post))
    Im_m_dot_t = np.hstack((Im_m_dot_t_pre, Im_m_dot_t_post))  # note "/dt" should be added

    T = np.hstack((t_pre, t_post))
    v_t = np.hstack((v_t_pre, v_t_post))
    Im_m_t = np.hstack((Im_m_t_pre, Im_m_t_post))
    Ca_HVA_m_inf_t = np.array([cf.Ca_HVA_m_inf(v_ti) for v_ti in v_t])
    Ca_HVA_h_inf_t = np.ones_like(v_t) * cf.Ca_HVA_h_inf(E_l)

    gCa_HVA_t = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf_t, Ca_HVA_h_inf_t])
    gIm_t = pf.calc_conductance('Im', gImbar, [Im_m_t])
    gl_pas_t = gl_pas * np.ones(np.size(v_t))

    I_Ca_HVA_t = pf.calc_current(v_t, gCa_HVA_t, E_Ca)
    I_Im_t = pf.calc_current(v_t, gIm_t, E_K)
    I_l_t = pf.calc_current(v_t, gl_pas_t, E_l)

    I_ext_per_area = pf.I_ext_constant(amp) / A

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

    data['v_pre'] = v_t_pre
    data['v_post'] = v_t_post
    data['Im_m_t_pre'] = Im_m_t_pre
    data['Im_m_t_post'] = Im_m_t_post
    data['t_pre'] = Im_m_t_pre
    data['t_post'] = Im_m_t_post


    dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init,
             timing_perturb, v_perturb_addition)] = data

    toc = time.time()
    print(
        'finished %f amp %f E_l %f mul_Ca %f mul_K %f V_init %f Im_m_init %f timing_perturb %f v_perturb_addition in %f secs' \
        % (amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb,
           v_perturb_addition, toc - tic))

#%%
# ========== plot ===========
plot_cons = [[0, -77, 5.1, 7.5, -30, 0, 10, 0],
             [0, -77, 5.1, 7.5, -30, 0, 5, 15],
             [0, -77, 5.1, 7.5, -30, 0, 5, -15],
             [0, -77, 5.1, 7.5, -30, 0, 10, -35],
             [0, -77, 5.1, 7.5, -30, 0, 15, -17],
             [0, -77, 5.1, 7.5, -30, 0, 15, -100]]
fig, (ax0) = plt.subplots(figsize=(4, 2))
for pair in condition_pairs:
    if pair in plot_cons:
        amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition = pair

        ax0.plot(dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['t'],
                 dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v'])

ax0.set_xlabel('ms')
ax0.set_ylabel('mV')
ax0.set_xlim([0, 40])
ax0.set_ylim([-120, 60])
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig7/fig7_spike_traces.svg", format='svg')

#%%
# ============= get spike properties ==============
props = {}
for pair in condition_pairs:
    amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition = pair
    props_curr = {}
    v_np = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v']

    start_t = 0

    # end_t
    logical_above_end_nexus_v = v_np > th_end_nexus_mV
    diff_logical_above_end_nexus_v = np.diff(logical_above_end_nexus_v)
    end_t = T[:-1][diff_logical_above_end_nexus_v][0]

    # duration
    ca_spike_dur = end_t - start_t

    # area
    [dummy, start_ind] = uf.find_nearest(T, start_t)
    [dummy, stop_ind] = uf.find_nearest(T, end_t)

    # max plateau value in specific t

    [dummy, ind_plat] = uf.find_nearest(T, t_plat)
    max_plat = v_np[ind_plat]

    props_curr['duration'] = ca_spike_dur
    props_curr['start_t'] = start_t
    props_curr['end_t'] = end_t
    props_curr['max_plat'] = max_plat

    props[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)] = props_curr

#%%
# =========== nullclines ==============
nclines_params = cfg_fig_7_1["nclines"]
Im_ms_low = nclines_params["Im_ms_low"]
Im_ms_high = nclines_params["Im_ms_high"]
Im_ms_delta = nclines_params["Im_ms_delta"]
Vs_low = nclines_params["Vs_low"]
Vs_high = nclines_params["Vs_high"]
Vs_delta = nclines_params["Vs_delta"]
n_samples = nclines_params["n_samples"]
scale = nclines_params["scale"]
dd = nclines_params["dd"]
ss = nclines_params["ss"]
lw = nclines_params["lw"]
dur_show_pert = nclines_params["dur_show_pert"]

samples_show_pert = int(dur_show_pert*(1/dt))

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

Ca_HVA_h_inf = cf.Ca_HVA_h_inf(E_l)
gCa_HVAbar = nexus_params['gCa_HVAbar_Ca_HVA'] * mul_Ca * TO_MS
gImbar = nexus_params['gImbar_Im'] * mul_K * TO_MS
gl_pas = nexus_params['g_pas'] * mul_gpas * TO_MS
I_ext_per_area = pf.I_ext_constant(amp) / A
v_ncline_param_dict = {
    'I_ext_per_area': I_ext_per_area,
    'gCa_HVAbar': gCa_HVAbar,
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


# draw nullclines
# ===============
fig, ax = plt.subplots(1, 1, figsize=(4, 4))

yIm_m_vnull = df.v_ncline(Vs, v_ncline_param_dict)
yIm_m_mnull = df.Im_m_ncline(Vs, Im_m_ncline_param_dict)

ax.plot(Vs[ind_singular + 1:], yIm_m_vnull[ind_singular + 1:],
        color=[0.6, 0.6, 0.6], label='v-null', linewidth=lw)
ax.plot(Vs, yIm_m_mnull,
        color=[0.4, 0.4, 0.4], linewidth=lw, label='Im_m-null')
ax.set_xlabel('mV', fontsize=16)
ax.set_ylabel('Im_m', fontsize=16)

# calc field arrows
x_dot = df.two_variables_model_constant_generation_calc_x_dot_all_pairs(x, amp,
                                                                        gCa_HVAbar, gImbar, gl_pas,
                                                                        E_Ca, E_K, E_l, A, Cm,
                                                                        shift, slope, k)

q = plt.quiver(x[::dd, 0],
               x[::dd, 1],
               x_dot[::dd, 0],
               x_dot[::dd, 1],
               pivot='mid',
               scale=scale,
               angles='xy',
               minlength=0.1)

# draw Ca2+ spike trajectory
for i, pair in enumerate(condition_pairs):
    amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition = pair

    # ctrl
    if Im_m_init == 0 and v_perturb_addition == 0:
        v_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v']
        Im_m_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t']
        xx = np.copy(v_t)[::ss]
        yy = np.copy(Im_m_t)[::ss]
        ax.plot(xx, yy, 'k', markersize=1, linewidth=lw, label='Ca Spike')

    if Im_m_init == 0.1 and v_perturb_addition == 0:
        v_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v']
        Im_m_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t']
        xx = np.copy(v_t)[::ss]
        yy = np.copy(Im_m_t)[::ss]
        ax.plot(xx, yy, 'k', markersize=1, linewidth=lw/2, label='Ca Spike')

    if Im_m_init == 0.2 and v_perturb_addition == 0:
        v_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v']
        Im_m_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t']
        xx = np.copy(v_t)[::ss]
        yy = np.copy(Im_m_t)[::ss]
        ax.plot(xx, yy, 'k', markersize=1, linewidth=lw/4, label='Ca Spike')

    # Termination
    elif timing_perturb == 2 and v_perturb_addition == -65:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post)[::ss]
        ax.plot(xx, yy, color='#fbc9be', markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 5 and v_perturb_addition == -45:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post)[::ss]
        ax.plot(xx, yy, color='#cd5c5c', markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 10 and v_perturb_addition == -35:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post)[::ss]
        ax.plot(xx, yy, color='#7c0a02', markersize=5, linewidth=lw, label='Ca Spike')

    # Conditional stabilization
    elif timing_perturb == 15 and v_perturb_addition == 5:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#fff192' ,markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 10 and v_perturb_addition == 8:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#ffea61' ,markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 5 and v_perturb_addition == 15:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#ffd400' ,markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 15 and v_perturb_addition == -17:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#D8AE48' ,markersize=5, linewidth=lw, label='Ca Spike')

    # Regeneration
    elif timing_perturb == 5 and v_perturb_addition == -15:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#1a936f' ,markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 10 and v_perturb_addition == -9:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#88d498' ,markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 15 and v_perturb_addition == -5:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#c6dabf' ,markersize=5, linewidth=lw, label='Ca Spike')

    # Reset
    elif timing_perturb == 5 and v_perturb_addition == -100:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,color ='#95c8d8' ,markersize=5, linewidth=lw, label='Ca Spike')
    elif timing_perturb == 15 and v_perturb_addition == -100:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:samples_show_pert ]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:samples_show_pert ]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy, color='#20639b', markersize=5, linewidth=lw, label='Ca Spike')

ax.set_xlim([-120, 50])
ax.set_ylim([0 - 0.01, 1 + 0.01])
if general_params["save_figs"]:
    fig.savefig("./Fig7/fig7_phase_plane_multi.svg", format='svg')

# ========== zoom in ========
nclines_zoom_params = cfg_fig_7_1["nclines_zoom"]
scale = nclines_zoom_params["scale"]
marks = nclines_zoom_params["marks"]
markew = nclines_zoom_params["markew"]
npoints = nclines_zoom_params["npoints"]

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
yIm_m_vnull = df.v_ncline(Vs, v_ncline_param_dict)
yIm_m_mnull = df.Im_m_ncline(Vs, Im_m_ncline_param_dict)

ax.plot(Vs[ind_singular + 1:], yIm_m_vnull[ind_singular + 1:],
        color=[0.6, 0.6, 0.6], label='v-null', linewidth=lw)
ax.plot(Vs, yIm_m_mnull,
        color=[0.4, 0.4, 0.4], linewidth=lw, label='Im_m-null')
ax.set_xlabel('mV', fontsize=16)
ax.set_ylabel('Im_m', fontsize=16)

x_dot = df.two_variables_model_constant_generation_calc_x_dot_all_pairs(x, amp,
                                                                        gCa_HVAbar, gImbar, gl_pas,
                                                                        E_Ca, E_K, E_l, A, Cm,
                                                                        shift, slope, k)

q = plt.quiver(x[::dd, 0],
               x[::dd, 1],
               x_dot[::dd, 0],
               x_dot[::dd, 1],
               pivot='mid',
               scale=scale,
               angles='xy',
               minlength=0.1)

# draw calcium spike trajectory
condition_pairs = [[amp, E_l, mul_Ca, mul_K, -30, 0, 10, 0],  # ctrl
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 5, 15],  # con. stab. exc
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 10, 8],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 15, 5],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 5, -15],  # regeneration
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 10, -9],
                   [amp, E_l, mul_Ca, mul_K, -30, 0, 15, -5]]

for i, pair in enumerate(condition_pairs):
    amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition = pair
    if Im_m_init == 0 and v_perturb_addition == 0:
        v_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v']
        Im_m_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t']
        xx = np.copy(v_t)[::ss]
        yy = np.copy(Im_m_t)[::ss]
        ax.plot(xx, yy, 'ok', markersize=1, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    if Im_m_init == 0.1 and v_perturb_addition == 0:
        v_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v']
        Im_m_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t']
        xx = np.copy(v_t)[::ss]
        yy = np.copy(Im_m_t)[::ss]
        ax.plot(xx, yy, 'k', markersize=1, linewidth=lw/2, label='Ca Spike')
    if Im_m_init == 0.2 and v_perturb_addition == 0:
        v_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v']
        Im_m_t = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t']
        xx = np.copy(v_t)[::ss]
        yy = np.copy(Im_m_t)[::ss]
        ax.plot(xx, yy, 'k', markersize=1, linewidth=lw/4, label='Ca Spike')

    # Termination
    elif timing_perturb == 2 and v_perturb_addition == -65:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post)[::ss]
        ax.plot(xx, yy, color='#fbc9be', markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')

    elif timing_perturb == 5 and v_perturb_addition == -45:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post)[::ss]
        ax.plot(xx, yy, color='#cd5c5c', markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    elif timing_perturb == 10 and v_perturb_addition == -35:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post)[::ss]
        ax.plot(xx, yy, color='#7c0a02', markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')

    # Conditional stabilization
    elif timing_perturb == 15 and v_perturb_addition == 5:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#fff192' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    elif timing_perturb == 10 and v_perturb_addition == 8:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#ffea61' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    elif timing_perturb == 5 and v_perturb_addition == 15:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#ffd400' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    elif timing_perturb == 15 and v_perturb_addition == -17:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#D8AE48' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')

    # Regeneration
    elif timing_perturb == 5 and v_perturb_addition == -15:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#1a936f' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    elif timing_perturb == 10 and v_perturb_addition == -9:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#88d498' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    elif timing_perturb == 15 and v_perturb_addition == -5:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#c6dabf' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')

    # Reset
    elif timing_perturb == 5 and v_perturb_addition == -100:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o',color ='#95c8d8' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')
    elif timing_perturb == 15 and v_perturb_addition == -100:
        v_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['v_post'][:npoints]
        Im_m_t_post = dic_log[(amp, E_l, mul_Ca, mul_K, V_init, Im_m_init, timing_perturb, v_perturb_addition)]['Im_m_t_post'][:npoints]
        xx = np.copy(v_t_post)[::ss]
        yy = np.copy(Im_m_t_post )[::ss]
        ax.plot(xx, yy,'o', color ='#20639b' ,markersize=marks, markeredgewidth=markew, markeredgecolor = 'k',linewidth=lw, label='Ca Spike')

ax.set_xlim([-25, 8])
ax.set_ylim([0.56, 0.67])
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig7/fig7_phase_plane_exc_zoom_in.svg", format='svg')
