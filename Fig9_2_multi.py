import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
import phys_funs as pf
import channel_funs as cf
import dynamics_funs as df
import analysis_funs as af
import util_funs as uf
from cfg import *
from constants import *

# =========== parameters ===========
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_params = pickle.load(file)
file.close()

general_params = cfg_fig_9["h"]
tmin = general_params["tmin"]
tmax = general_params["tmax"]
dt = general_params["dt"]
pert_type = general_params["pert_type"]
T = np.arange(tmin, tmax, dt)

phys_params = cfg_fig_9["phys"]
shift = phys_params["shift"]
slope = phys_params["slope"]
k = phys_params["k"]
gK = phys_params["gK"]
gNa = phys_params["gNa"]
gL = phys_params["gL"]
g_con = phys_params["g_con"]
A_soma = phys_params["A_soma"]
A_nexus = phys_params["A_nexus"]
Cm_soma = phys_params["Cm_soma"]
Cm_nexus = phys_params["Cm_nexus"]
E_K = phys_params["E_K"]
E_Na = phys_params["E_Na"]
E_l = phys_params["E_l"]
E_Ca = phys_params["E_Ca"]

sim_params = cfg_fig_9["sim"]
amp_soma = sim_params["amp_soma"]
amp_nexus = sim_params["amp_nexus"]
Im_m_init = sim_params["Im_m_init"]
mul_gpas = sim_params["mul_gpas"]
mul_K = sim_params["mul_K"]
amp_nexus_syn_gen = sim_params["amp_nexus_syn_gen"]
ts_nexus_gen = sim_params["ts_nexus_gen"]
tau1_nexus_gen = sim_params["tau1_nexus_gen"]
tau2_nexus_gen = sim_params["tau2_nexus_gen"]
tau1_nexus_pert = sim_params["tau1_nexus_pert"]
tau2_nexus_pert = sim_params["tau2_nexus_pert"]
V_soma_init = sim_params["V_soma_init"]
V_nexus_init = sim_params["V_nexus_init"]

ts_nexus_perts = cfg_fig_9_2["sim"]["ts_nexus_pert"]
amp_nexus_syn_perts = cfg_fig_9_2["sim"]["amp_nexus_syn_pert"]
mul_Cas = cfg_fig_9_2["sim"]["mul_Ca"]

# ============= run simulation ==============
dic_log = {}

tic = time.time()
for mul_Ca in mul_Cas:
    for amp_nexus_syn_pert in amp_nexus_syn_perts:
        for ts_nexus_pert in ts_nexus_perts:

            tic = time.time()
            data = {}

            # State vector parameters: v, all gates
            gCa_HVAbar = nexus_params['gCa_HVAbar_Ca_HVA'] * mul_Ca * TO_MS
            gImbar = nexus_params['gImbar_Im'] * mul_K * TO_MS
            gl_pas = nexus_params['g_pas'] * mul_gpas * TO_MS

            Y = np.array([V_soma_init, cf.HH_K_n_inf(V_soma_init),
                          cf.HH_Na_m_inf(V_soma_init),
                          cf.HH_Na_h_inf(V_soma_init),
                          V_nexus_init, Im_m_init])

            # Solve ODE system
            Vy = odeint(df.two_compartments_model_with_perturbation, Y, T,
                        args=(amp_soma, amp_nexus, amp_nexus_syn_gen, ts_nexus_gen, tau1_nexus_gen, tau2_nexus_gen,
                              amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert, tau2_nexus_pert,
                              gCa_HVAbar, gImbar, gl_pas, gK, gNa, gL, g_con,
                              E_Ca, E_K, E_l, E_Na, A_soma, A_nexus, Cm_soma, Cm_nexus, shift, slope, k))

            dy_list = []
            for i in range(len(T)):
                dy = df.two_compartments_model_with_perturbation(
                    Vy[i], T[i], amp_soma, amp_nexus, amp_nexus_syn_gen, ts_nexus_gen, tau1_nexus_gen, tau2_nexus_gen,
                    amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert, tau2_nexus_pert,
                    gCa_HVAbar, gImbar, gl_pas, gK, gNa, gL, g_con,
                    E_Ca, E_K, E_l, E_Na, A_soma, A_nexus, Cm_soma, Cm_nexus, shift, slope, k)
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
            Ca_HVA_h_inf_t = np.ones_like(v_nexus_t) * cf.Ca_HVA_h_inf(E_l)
            Ca_HVA_m_inf_t = np.array([cf.Ca_HVA_m_inf(v_ti) for v_ti in v_nexus_t])  # inst
            gCa_HVA_t = pf.calc_conductance('HVA_first_order', gCa_HVAbar, [Ca_HVA_m_inf_t, Ca_HVA_h_inf_t])
            gIm_t = pf.calc_conductance('Im', gImbar, [Im_m_t])
            gl_pas_t = gl_pas * np.ones(np.size(v_nexus_t))

            I_Ca_HVA_t = pf.calc_current(v_nexus_t, gCa_HVA_t, E_Ca)
            I_Im_t = pf.calc_current(v_nexus_t, gIm_t, E_K)
            I_l_nexus_t = pf.calc_current(v_nexus_t, gl_pas_t, E_l)

            # soma
            GK_t = gK * np.power(n_t, 4.0)
            GNa_t = gNa * np.power(m_t, 3.0) * h_t
            GL_t = gL * np.ones(np.size(v_soma_t))

            IK_t = pf.calc_current(v_soma_t, GK_t, E_K)
            INa_t = pf.calc_current(v_soma_t, GNa_t, E_Na)
            I_l_soma_t = pf.calc_current(v_soma_t, GL_t, E_l)

            # Input stimulus
            I_ext_nexus_per_area = pf.I_ext_constant(amp_nexus) / A_nexus
            I_ext_nexus_syn_gen_vec = np.array([pf.I_ext_syn(t_curr, amp_nexus_syn_gen, ts_nexus_gen,
                                                             tau1_nexus_gen, tau2_nexus_gen) for t_curr in T])
            I_ext_nexus_syn_pert_vec = np.array([pf.I_ext_syn(t_curr, amp_nexus_syn_pert, ts_nexus_pert,
                                                              tau1_nexus_pert, tau2_nexus_pert) for t_curr in T])
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

            dic_log[(mul_Ca, amp_nexus_syn_pert,ts_nexus_pert)] = data

            toc = time.time()
            print(f'finished (amp_nexus_syn_pert {amp_nexus_syn_pert}, ts_nexus_pert {ts_nexus_pert}) '
                  f'in {toc - tic} seconds')


# =========== get spike properties ===========
props = {}

for mul_Ca in mul_Cas:
    for amp_nexus_syn_pert in amp_nexus_syn_perts:
        for ts_nexus_pert in ts_nexus_perts:
            props_curr = {}

            soma_v_np = dic_log[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['v_soma_t']
            nexus_v_np = dic_log[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['v_nexus_t']

            (log, start_t, stop_t) = af.is_calcium_spike(T, nexus_v_np)
            props_curr['log'] = log

            if log:
                ca_spike_dur = stop_t - start_t

                [dummy, start_ind] = uf.find_nearest(T, start_t)
                [dummy, stop_ind] = uf.find_nearest(T, stop_t)

                n_sp = af.n_spikes(soma_v_np)

                props_curr['duration'] = ca_spike_dur
                props_curr['n_sp'] = n_sp
                props_curr['start_t'] = start_t
                props_curr['stop_t'] = stop_t


            props[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)] = props_curr


# ===== number of spikes ========
sp_matrix = np.zeros((amp_nexus_syn_perts.size, ts_nexus_perts.size))
for mul_Ca in mul_Cas:
    for i, amp_nexus_syn_pert in enumerate(amp_nexus_syn_perts):
        for j, ts_nexus_pert in enumerate(ts_nexus_perts):
            if props[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['log'] == 1:
                sp_matrix[i,j] = props[(mul_Ca, amp_nexus_syn_pert, ts_nexus_pert)]['n_sp']


fig, ax = plt.subplots(figsize=(2, 2))
ax = sns.heatmap(sp_matrix, cmap='gray', cbar=False)
ax.invert_yaxis()
ax.set_xticks(np.arange(0,ts_nexus_perts.size,2))
ax.set_yticks(np.arange(0,amp_nexus_syn_perts.size, 2))
ax.set_xticklabels(ts_nexus_perts[::2]-ts_nexus_gen)
ax.set_yticklabels(amp_nexus_syn_perts[::2])
ax.set_ylabel('amp_nexus_syn_pert_list')
ax.set_xlabel('pert onsets')
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig9/Fig9_heatmap.svg", format='svg')


