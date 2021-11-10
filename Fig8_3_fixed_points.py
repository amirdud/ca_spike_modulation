import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from scipy.integrate import odeint
import mod_dynamics_funs as mdf
import funs
import sympy as sp

# ============= parameters ===========
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_parameters = pickle.load(file)
file.close()

# Set random seed (for reproducibility)
np.random.seed(1000)

# Start and end time (in milliseconds)
tmin = 0.0
tmax = 500
dt = 0.025
T = np.arange(tmin, tmax, dt)

# general parameters
to_mS = 1000
shift = -39
slope = 2.3
k =2.0e-3

E_K = -85  # mV
E_Ca = 120  # mV
E_l = -77  # mV
Cm = 1.0  # uF/cm2

A = 1e-5  # cm2 (18 um cell diameter)

mul_gpas = 1
V_init = -30
Im_m_init = 0

# pert_type = 'ACh'
# pert_type  = 'constant_current'
pert_type  = 'ACh_K'

if pert_type == 'ACh':
    mul_Cas_diff = 0.1

    amps = np.array([0])
    mul_Cas = np.round(np.arange(5.1, 9 + mul_Cas_diff, mul_Cas_diff), 2)
    mul_Ks = np.array([7.5])

elif pert_type == 'constant_current':
    amps_diff = 0.00001

    amps = np.round(np.arange(0, 0.00034 + amps_diff, amps_diff), 5)
    # amps = [0, 0.00015, 0.0003]

    mul_Cas = [5.1]
    mul_Ks = np.array([7.5])

elif pert_type == 'ACh_K':
    mul_Ks_diff = -0.1

    amps = np.array([0])
    mul_Cas = np.array([5.1])
    mul_Ks = np.round(np.arange(7.5, 4 + mul_Ks_diff, mul_Ks_diff), 2)

else:
    raise ValueError(f"{pert_type} is invalid input for modulation_type ")


def I_ext(amp):
    return amp


def compute_derivatives(y, t0,amp):
    dy = np.zeros((2,))

    v         = y[0]
    Im_m      = y[1]

    Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
    Ca_HVA_m_inf = mdf.Ca_HVA_m_inf(v) # inst
    gCa_HVA = mdf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])

    gIm = mdf.calc_conductance('Im', gImbar, [Im_m])

    I_Ca_HVA = mdf.calc_current(v,gCa_HVA,E_Ca)
    I_Im = mdf.calc_current(v, gIm, E_K)
    I_l = mdf.calc_current(v, gl_pas, E_l)
    I_I_ext = I_ext(amp) / A

    # dv/dt
    dy[0] = (I_I_ext - I_Ca_HVA - I_Im - I_l) / Cm

    # Im dm/dt
    alpha,beta = mdf.Im_m_alpha_beta_modified(v,shift,slope,k)
    dy[1] = (alpha * (1.0 - Im_m)) - (beta * Im_m)

    return dy


# ================= run simulation =============
dic_log = {}
for amp in amps:
    for mul_Ca in mul_Cas:
        for mul_K in mul_Ks:

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

            I_ext_per_area = I_ext(amp) / A

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

            dic_log[(amp, mul_Ca, mul_K)] = data

            toc = time.time()
            print('finished %f amp %f mul_Ca %f mul_K in %f secs' % (amp, mul_Ca, mul_K, toc - tic))

# =============== plot =================
from matplotlib import cm

evenly_spaced_interval = np.linspace(0, 1, 7)
colors = [cm.coolwarm(k) for k in evenly_spaced_interval]

fig, (ax0) = plt.subplots(1,1,figsize=(2.5, 1.1))

for amp in amps:
    for mul_Ca in mul_Cas:
        for mul_K in mul_Ks:
            ax0.plot(dic_log[(amp, mul_Ca, mul_K)]['t'],
                     dic_log[(amp, mul_Ca, mul_K)]['v'], 'k')


ax0.set_ylabel('Vm (mV)',fontsize = 16)
ax0.set_xlim([0,500])
ax0.set_ylim([-100,50])
ax0.set_xticks([0,250,500])
plt.show()
# fig.savefig("./Fig8/fig8_spiking_constant_current_R3.svg", format='svg')
# fig.savefig("./Fig8/fig8_spiking_ACh_R2.svg", format='svg')


# ========= nullclines ===========
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

Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
gl_pas = nexus_parameters['g_pas'] * mul_gpas * to_mS
v_ncline_param_dict = {
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

fig, ax = plt.subplots(figsize=(2.5, 2.5))
for amp in amps:
    for mul_Ca in mul_Cas:
        for mul_K in mul_Ks:
            gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
            gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
            I_ext_per_area = I_ext(amp) / A
            v_ncline_param_dict['I_ext_per_area'] = I_ext_per_area
            v_ncline_param_dict['gCa_HVAbar'] = gCa_HVAbar
            v_ncline_param_dict['gImbar'] = gImbar

            yIm_m_vnull = funs.v_ncline(Vs, v_ncline_param_dict)
            yIm_m_mnull = funs.Im_m_ncline(Vs, Im_m_ncline_param_dict)

            # draw nullclines
            ax.plot(Vs[ind_singular + 1:], yIm_m_vnull[ind_singular + 1:],
                    color=[0.6, 0.6, 0.6], label='v-null', linewidth=lw)
            ax.plot(Vs, yIm_m_mnull,
                    color=[0.4, 0.4, 0.4], linewidth=lw, label='Im_m-null')

            # draw calcium spike trajectory
            v_t = dic_log[(amp, mul_Ca, mul_K)]['v']
            Im_m_t = dic_log[(amp, mul_Ca, mul_K)]['Im_m_t']
            xx = np.copy(v_t)[::ss]
            yy = np.copy(Im_m_t)[::ss]
            ax.plot(xx, yy, 'k', label='Ca Spike', linewidth=lw)

ax.set_xlim([-120, 50])
ax.set_ylim([0 - 0.01, 1 + 0.01])
ax.set_xlabel('mV',fontsize=16)
ax.set_ylabel('Im_m',fontsize=16)

plt.show()
# fig.savefig("./Fig8/fig8_phase_plane_constant_current_R3.svg", format='svg')
# fig.savefig("./Fig8/fig8_phase_plane_ACh_R2.svg", format='svg')
# fig.savefig("./Fig8/fig8_phase_plane_ACh_3_conditions.svg", format='svg')


# ============ fixed points =============
props_list_mul_K = []
props_list_mul_Ca = []
props_list_fixed_points_V = []
props_list_fixed_points_m = []

props = {}
for amp in amps:
    for mul_Ca in mul_Cas:
        for mul_K in mul_Ks:
            props_curr = {}

            gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
            gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS
            I_ext_per_area = I_ext(amp) / A
            Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
            gl_pas = nexus_parameters['g_pas'] * to_mS
            v_ncline_param_dict['I_ext_per_area'] = I_ext_per_area
            v_ncline_param_dict['gCa_HVAbar'] = gCa_HVAbar
            v_ncline_param_dict['gImbar'] = gImbar

            yIm_m_vnull = funs.v_ncline(Vs, v_ncline_param_dict)
            yIm_m_mnull = funs.Im_m_ncline(Vs, Im_m_ncline_param_dict)

            # find bifuractions
            f1 = yIm_m_vnull[ind_singular + 1:]
            g1 = yIm_m_mnull[ind_singular + 1:]
            fixed_points_inds = funs.find_graph_intersections(f1, g1)
            fixed_points_V = Vs[ind_singular + 1:][fixed_points_inds]
            fixed_points_m = g1[fixed_points_inds]
            props_curr['fixed_points_V'] = fixed_points_V
            props_curr['fixed_points_m'] = fixed_points_m

            props[(amp, mul_Ca, mul_K)] = props_curr

            props_list_mul_Ca.append(mul_Ca)
            props_list_mul_K.append(mul_K)
            props_list_fixed_points_V.append(fixed_points_V)
            props_list_fixed_points_m.append(fixed_points_m)

# ============== stability analysis ===============
v, m = sp.symbols('v, m', real=True)

Ca_HVA_h_inf = mdf.Ca_HVA_h_inf(E_l)
Ca_HVA_m_inf = mdf.Ca_HVA_m_inf_sympy(v)  # inst
I_l = mdf.calc_current(v, gl_pas, E_l)

fig, ax = plt.subplots(figsize=(4, 2))
fp_colors = [[1, 1, 1], [0, 0, 0]]
for amp in amps:
    for mul_Ca in mul_Cas:
        if len(mul_Cas)>1:
            J_list = []
            stbl_str_list = []
            stbl_list = []
        for mul_K in mul_Ks:
            if len(mul_Ks) > 1:
                J_list = []
                stbl_str_list = []
                stbl_list = []

            # build sympy symbolic expression
            gCa_HVAbar = nexus_parameters['gCa_HVAbar_Ca_HVA'] * mul_Ca * to_mS
            gImbar = nexus_parameters['gImbar_Im'] * mul_K * to_mS

            gCa_HVA = mdf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf, Ca_HVA_h_inf])
            gIm = mdf.calc_conductance('Im', gImbar, [m])

            I_ext_per_area = I_ext(amp) / A
            I_Ca_HVA = mdf.calc_current(v, gCa_HVA, E_Ca)
            I_Im = mdf.calc_current(v, gIm, E_K)

            dvdt = (I_ext_per_area - I_Ca_HVA - I_Im - I_l) / Cm
            dmdt = ((k * sp.exp(slope * 0.04 * (v - shift))) * (1.0 - m)) - (
                    (k * sp.exp(-slope * 0.04 * (v - shift))) * m)

            dfdv = sp.diff(dvdt, v)
            dfdm = sp.diff(dvdt, m)
            dgdv = sp.diff(dmdt, v)
            dgdm = sp.diff(dmdt, m)

            # print('df/dv = ', dfdv)
            # print('df/dm = ', dfdm)
            # print('dg/dv = ', dgdv)
            # print('dg/dm = ', dgdm)

            # evaluate
            equilibria_v = props[(amp, mul_Ca, mul_K)]['fixed_points_V']
            equilibria_m = props[(amp, mul_Ca, mul_K)]['fixed_points_m']

            for eq_v, eq_m in zip(equilibria_v, equilibria_m):
                # Jacobian:
                J = np.zeros((2, 2))

                J[0, 0] = eval(str(dfdv).replace("v", "eq_v").replace("m", "eq_m"))
                J[0, 1] = eval(str(dfdm).replace("v", "eq_v").replace("m", "eq_m"))
                J[1, 0] = eval(str(dgdv).replace("v", "eq_v").replace("m", "eq_m"))
                J[1, 1] = eval(str(dgdm).replace("v", "eq_v").replace("m", "eq_m"))
                J_list.append(J)

                # Stability:
                [eig_vals, eig_vecs] = np.linalg.eig(J)
                J_det = np.linalg.det(J)
                J_tr = np.trace(J)

                # imaginary eig values
                if isinstance(eig_vals[0], complex) and isinstance(eig_vals[1], complex):
                    if eig_vals[0].real < 0 and eig_vals[1].real < 0:
                        stbl_str = 'stable focus'
                        stbl = 1

                    elif eig_vals[0].real > 0 and eig_vals[1].real > 0:
                        stbl_str = 'unstable focus'
                        stbl = 0

                # real eig values
                elif not isinstance(eig_vals[0], complex) and not isinstance(eig_vals[1], complex):
                    if eig_vals[0] < 0 and eig_vals[1] < 0:
                        stbl_str = 'stable node'
                        stbl = 1

                    elif eig_vals[0].real > 0 and eig_vals[1].real > 0:
                        stbl_str = 'unstable node'
                        stbl = 0

                    else:
                        stbl_str = 'saddle'
                        stbl = 0

                else:
                    raise ValueError('Something is wrong with the eigen values')

                print(f"amp: {amp}; mul_Ca: {mul_Ca/mul_Cas[0]}")
                print(f"stability: {stbl_str}")
                # print('{:.2f} mV: '.format(eq_v), stbl_str)
                # print('eig_vals: ', eig_vals)
                stbl_str_list.append(stbl_str)
                stbl_list.append(stbl)

            props[(amp, mul_Ca, mul_K)]['J_list'] = J_list
            props[(amp, mul_Ca, mul_K)]['stbl_str_list'] = stbl_str_list
            props[(amp, mul_Ca, mul_K)]['stbl_list'] = stbl_list

            if pert_type == 'ACh':
                stbl_list = np.array(props[(amp, mul_Ca, mul_K)]['stbl_list'])
                colors_list = [fp_colors[k] for k in stbl_list]
                ax.scatter(mul_Ca/mul_Cas[0]*np.ones_like(props[(amp, mul_Ca, mul_K)]['stbl_list']),
                           props[(amp, mul_Ca, mul_K)]['fixed_points_V'],c=colors_list,s=25,edgecolors='k')
                ax.set_xlabel('gCaHVA multiplier')

            elif pert_type == 'constant_current':
                stbl_list = np.array(props[(amp, mul_Ca, mul_K)]['stbl_list'])
                colors_list = [fp_colors[k] for k in stbl_list]
                ax.scatter(amp * np.ones_like(props[(amp, mul_Ca, mul_K)]['stbl_list']),
                           props[(amp, mul_Ca, mul_K)]['fixed_points_V'], c=colors_list, s=25, edgecolors='k')
                ax.set_xlabel('current')

            elif pert_type == 'ACh_K':
                stbl_list = np.array(props[(amp, mul_Ca, mul_K)]['stbl_list'])
                colors_list = [fp_colors[k] for k in stbl_list]
                ax.scatter(mul_K / mul_Ks[0] * np.ones_like(props[(amp, mul_Ca, mul_K)]['stbl_list']),
                           props[(amp, mul_Ca, mul_K)]['fixed_points_V'], c=colors_list, s=25, edgecolors='k')
                ax.set_xlabel('gIm multiplier')

ax.set_ylabel('mV')
plt.show()
# fig.savefig("./Fig8/fig8_fixed_points_constant_current_R3.svg", format='svg')
# fig.savefig("./Fig8/fig8_fixed_points_ACh_R2.svg", format='svg')

