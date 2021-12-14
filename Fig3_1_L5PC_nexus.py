
import matplotlib.pyplot as plt
from matplotlib import cm
import analysis_funs as af
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from cfg import *
from constants import *

general_params = cfg_fig_3_1["h"]
nf.define_nrn_general_params(general_params)

# ====== model ======
L5PC = nf.get_L5PC_model()

# ========== generation ==========
nexus = nf.get_nexus(L5PC)
syn_gen_nexus_params = cfg_fig_3_1["syn_gen_nexus"]
syn_gen_nexus = nf.epsc_like_current_insert(nexus)
nf.epsc_like_current_add_params(syn_gen_nexus, syn_gen_nexus_params)

# ============== simulation =================
stims = {'syn_gen_nexus': syn_gen_nexus}
var = L5PC_define_vars(stims=stims)
var = L5PC_record(var, L5PC, stims=stims)
run_nrn_simulation()
simulation_data = uf.convert_var_to_numpy(var)

# ======== order currents and conductances ========
current_names_list = ["I_CaHVA", "I_CaLVA", "I_SK", "I_SKv3", "I_Im", "I_Ih", "I_NaTs2_t"]
conductance_names_list = ["g_CaHVA", "g_CaLVA", "g_SK", "g_SKv3", "g_Im", "g_Ih", "g_NaTs2_t"]

current_list = []
conductance_list = []
for (current_name, conductance_name) in zip(current_names_list, conductance_names_list):
    current = simulation_data["nexus"][current_name]
    conductance = simulation_data["nexus"][conductance_name]
    current_list.append(current)
    conductance_list.append(conductance)


max_vals_sorted_inds = af.order_currents(current_list, current_names_list)
current_list_sorted = [current_list[i] for i in max_vals_sorted_inds]
current_names_list_sorted = [current_names_list[i] for i in max_vals_sorted_inds]
conductance_list_sorted = [conductance_list[i] for i in max_vals_sorted_inds]
conductance_names_list_sorted = [conductance_names_list[i] for i in max_vals_sorted_inds]

# ======== plot ========
evenly_spaced_interval = np.linspace(0, 1, 7)
colors = [cm.coolwarm(k) for k in evenly_spaced_interval]

t_shifted = simulation_data["t"] - syn_gen_nexus_params['onset']
fig, ax0 = plt.subplots(figsize=(4, 4))
for i, current in enumerate(current_list_sorted):
    ax0.plot(t_shifted, current, label=current_names_list_sorted[i], color=colors[i])

ax0.set_ylabel('mA/cm2', fontsize=16)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.legend(loc='lower right')
ax0.set_ylim([-0.1, 0.15])
ax0.set_xlim([-10, 70])
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig3/fig3_L5PC_currents.svg", format='svg')


fig,ax1 = plt.subplots(figsize=(4,2))
ax1.plot(t_shifted, simulation_data["soma"]["v"], SOMA_STYLE, label='soma')
ax1.plot(t_shifted, simulation_data["nexus"]["v"], NEXUS_STYLE, label='nexus')
ax1.set_ylabel('mV', fontsize=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_ylim([-90, 50])
ax1.set_xlim([-10, 70])
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig3/fig3_L5PC_spike.svg", format='svg')

fig, ax2 = plt.subplots(figsize=(4, 0.5))
ax2.plot(t_shifted, simulation_data["stims"]["I_syn_gen_nexus"] * (-1), NEXUS_STYLE, label='I')
ax2.set_ylabel('nA', fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim([0, 2.5])
ax2.set_xlim([-10, 70])
ax2.set_xlabel('ms', fontsize=16)
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig3/fig3_L5PC_epsp.svg", format='svg')

fig, ax3 = plt.subplots(figsize=(4, 4))
for i, conductance in enumerate(conductance_list_sorted):
    ax3.plot(t_shifted, conductance, label=conductance_names_list_sorted[i], color=colors[i])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.set_xlim([-10, 70])
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig3/fig3_L5PC_conductances.svg", format='svg')
