import time
import pickle
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.integrate import odeint
import phys_funs as pf
import channel_funs as cf
import dynamics_funs as df
from cfg import *
from constants import *

# =========== parameters ===========
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_params = pickle.load(file)
file.close()

general_params = cfg_fig_8["h"]
tmin = general_params["tmin"]
tmax = general_params["tmax"]
dt = general_params["dt"]
T = np.arange(tmin, tmax, dt)

phys_params = cfg_fig_8["phys"]
E_K = phys_params["E_K"]
E_Ca = phys_params["E_Ca"]
E_l = phys_params["E_l"]
Cm = phys_params["Cm"]
A = phys_params["A"]
shift = phys_params["shift"]
slope = phys_params["slope"]
k = phys_params["k"]

sim_params = cfg_fig_8["sim"]
mul_K = sim_params["mul_K"]
mul_gpas = sim_params["mul_gpas"]
V_init = sim_params["V_init"]

amp = cfg_fig_8_1["sim"]["amp"]
mul_Cas = cfg_fig_8_1["sim"]["mul_Ca"]
Im_m_init = cfg_fig_8_1["sim"]["Im_m_init"]

# =============== simulation ============
dic_log = {}
for mul_Ca in mul_Cas:
    tic = time.time()
    data = {}

    gCa_HVAbar = nexus_params['gCa_HVAbar_Ca_HVA'] * mul_Ca * TO_MS
    gImbar = nexus_params['gImbar_Im'] * mul_K * TO_MS
    gl_pas = nexus_params['g_pas'] * mul_gpas * TO_MS

    # State vector parameters: v, all gates
    Y = np.array([V_init, Im_m_init])

    # Solve ODE system
    Vy = odeint(df.compute_derivatives_two_variables_model_constant_generation,
                Y, T, args=(amp, gCa_HVAbar, gImbar, gl_pas, E_Ca, E_K, E_l, A, Cm, shift, slope, k))

    # get derivatives
    dy_list = []
    for i in range(len(T)):
        dy = df.compute_derivatives_two_variables_model_constant_generation(
            Vy[i], T[i], amp, gCa_HVAbar, gImbar, gl_pas, E_Ca, E_K, E_l, A, Cm, shift, slope, k)
        dy_list.append(dy)
    Vdy = np.array(dy_list)
    v_dot_t = Vdy[:, 0]
    Im_m_dot_t = Vdy[:, 1]

    v_t = Vy[:, 0]
    Im_m_t = Vy[:, 1]

    Ca_HVA_h_inf_t = np.ones_like(v_t) * cf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf_t = np.array([cf.Ca_HVA_m_inf(v_ti) for v_ti in v_t])  # inst
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

    dic_log[mul_Ca] = data

    toc = time.time()
    print('finished %f mul_Ca  in %f secs' % (mul_Ca, toc - tic))

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
if general_params["save_figs"]:
    fig.savefig("./Fig8/fig8_add_ACh_spike.svg", format='svg')


# ========== nullclines ===========
nclines_params = cfg_fig_8["nclines"]
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

Im_ms = np.arange(Im_ms_low, Im_ms_high + Im_ms_delta, Im_ms_delta)
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
gImbar = nexus_params['gImbar_Im'] * mul_K * TO_MS
gl_pas = nexus_params['g_pas'] * mul_gpas * TO_MS
I_ext_per_area = pf.I_ext_constant(amp) / A
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


# draw nullclines
# ===============
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for j, mul_Ca in enumerate(mul_Cas):
    gCa_HVAbar = nexus_params['gCa_HVAbar_Ca_HVA'] * mul_Ca * TO_MS
    v_ncline_param_dict['gCa_HVAbar'] = gCa_HVAbar
    yIm_m_vnull = df.v_ncline(Vs, v_ncline_param_dict)
    yIm_m_mnull = df.Im_m_ncline(Vs, Im_m_ncline_param_dict)

    # draw nullclines
    ax.plot(Vs[ind_singular + 1:], yIm_m_vnull[ind_singular + 1:],
            color=colors_reds[j], label='v-null', linewidth=lw)
    ax.plot(Vs, yIm_m_mnull,
            color=[0.4, 0.4, 0.4], linewidth=lw, label='Im_m-null')
    ax.set_xlabel('mV', fontsize=16)
    ax.set_ylabel('Im_m', fontsize=16)

    # calc field arrows
    x_dot = df.two_variables_model_constant_generation_calc_x_dot_all_pairs(x, amp,
                                                                            gCa_HVAbar, gImbar, gl_pas,
                                                                            E_Ca, E_K, E_l, A, Cm,
                                                                            shift, slope, k)

    # draw calcium spike trajectory
    v_t = dic_log[mul_Ca]['v']
    Im_m_t = dic_log[mul_Ca]['Im_m_t']
    xx = np.copy(v_t)[::ss]
    yy = np.copy(Im_m_t)[::ss]
    ax.plot(xx, yy, color=colors_greens[j], markersize=1, linewidth=lw, label='Ca Spike')

    ax.set_xlim([-120, 50])
    ax.set_ylim([0 - 0.01, 1 + 0.01])

plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig8/fig8_phase_plane_add_ACh.svg", format='svg')

