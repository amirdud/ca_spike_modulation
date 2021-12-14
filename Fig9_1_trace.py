import time
import pickle
import matplotlib.pyplot as plt
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
amp_nexus_syn_pert = sim_params["pert_dependent_params"][pert_type]["amp_nexus_syn_pert"]
mul_Ca = sim_params["pert_dependent_params"][pert_type]["mul_Ca"]

ts_nexus_pert = cfg_fig_9_1["sim"]["ts_nexus_pert"]

# ============= run simulation ==============
dic_log = {}
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
            args=(amp_soma, amp_nexus, amp_nexus_syn_gen, ts_nexus_gen,tau1_nexus_gen, tau2_nexus_gen,
                  amp_nexus_syn_pert, ts_nexus_pert, tau1_nexus_pert, tau2_nexus_pert,
                  gCa_HVAbar, gImbar, gl_pas, gK, gNa, gL, g_con,
                  E_Ca, E_K, E_l, E_Na, A_soma, A_nexus, Cm_soma, Cm_nexus, shift, slope, k))

dy_list = []
for i in range(len(T)):
    dy = df.two_compartments_model_with_perturbation(
        Vy[i], T[i], amp_soma, amp_nexus, amp_nexus_syn_gen, ts_nexus_gen,tau1_nexus_gen, tau2_nexus_gen,
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

data['v_soma_dot_t'] = v_soma_dot_t
data['n_dot_t'] = n_dot_t
data['m_dot_t'] = m_dot_t
data['h_dot_t'] = h_dot_t
data['v_nexus_dot_t'] = v_nexus_dot_t
data['Im_m_dot_t'] = Im_m_dot_t

dic_log[(amp_soma, amp_nexus)] = data

toc = time.time()
print(f'finished ({amp_soma}, {amp_nexus})')


# ================= plot ======================
color_soma = np.array([85, 149, 136]) / 255
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(1.2, 2))
ax0.plot(dic_log[(amp_soma, amp_nexus)]['t'], dic_log[(amp_soma, amp_nexus)]['v_nexus_t'], 'k', linewidth=1)
ax1.plot(dic_log[(amp_soma, amp_nexus)]['t'], dic_log[(amp_soma, amp_nexus)]['v_soma_t'], color=color_soma, linewidth=1)
if amp_nexus_syn_pert != 0:
    ax0.plot([ts_nexus_pert, ts_nexus_pert], [-50, 40])
ax0.set_xlim([45, 100])
ax1.set_xlim([45, 100])
plt.show()
if general_params["save_figs"]:
    if pert_type == 'ctrl':
        fig.savefig("./Fig9/Fig9_ctrl_trace.svg", format='svg')
    if pert_type == 'ipsp':
        fig.savefig("./Fig9/Fig9_inh_trace.svg", format='svg')
    if pert_type == 'epsp':
        fig.savefig("./Fig9/Fig9_exc_trace.svg", format='svg')
    if pert_type == 'ACh':
        fig.savefig("./Fig9/Fig9_ACh_trace.svg", format='svg')
