
import matplotlib.pyplot as plt
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from cfg import *
from constants import *

general_params = cfg_fig_1_1["h"]
nf.define_nrn_general_params(general_params)

# ====== model ======
L5PC = nf.get_L5PC_model()

# ====== generation =====
nexus = nf.get_nexus(L5PC)
syn_nexus_params = cfg_fig_1_1["syn_nexus"]
syn_nexus = nf.epsc_like_current_insert(nexus)
nf.epsc_like_current_add_params(syn_nexus, syn_nexus_params)

# ========= simulation ========
stims = {'syn_nexus': syn_nexus}
var = L5PC_define_vars(stims=stims)
var = L5PC_record(var, L5PC, stims=stims)
run_nrn_simulation()
simulation_data = uf.convert_var_to_numpy(var)

# ======== plot =========
var['t_shifted'] = uf.get_shifted_time(var['t'], syn_nexus_params["onset"])
f, ax1 = plt.subplots(figsize=(4, 2))
ax1.plot(var['t_shifted'], var['soma']['v'], SOMA_STYLE, label='soma_v')
ax1.plot(var['t_shifted'], var['nexus']['v'], NEXUS_STYLE, label='nexus_v')
ax1.set_ylabel('mV', fontsize=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_ylim([-90, 50])
ax1.set_xlim([-10, 70])

plt.show()
if general_params["save_figs"]:
    f.savefig("./Fig1/fig1_EPSP_Ca_Spike.svg", format='svg')

f, ax2 = plt.subplots(figsize=(4, 0.5))
ax2.plot(var['t_shifted'], var['stims']['I_syn_nexus']*(-1), NEXUS_STYLE, label='I_syn_nexus')
ax2.set_ylabel('nA', fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim([0, 2.5])
ax2.set_xlim([-10, 70])
ax2.set_xlabel('ms', fontsize=16)

plt.show()
if general_params["save_figs"]:
    f.savefig("./Fig1/fig1_EPSP_current.svg", format='svg')
