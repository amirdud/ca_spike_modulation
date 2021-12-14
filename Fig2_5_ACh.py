import time
from neuron import h,gui
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import analysis_funs as af
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from constants import *
from cfg import *

general_params = cfg_fig_2_5["h"]
nf.define_nrn_general_params(general_params)

# ======= model =======
L5PC = nf.get_L5PC_model()

# ========== generation ==========
nexus = nf.get_nexus(L5PC)
syn_gen_nexus_params = cfg_fig_2_5["syn_gen_nexus"]
syn_gen_nexus = nf.epsc_like_current_insert(nexus)
nf.epsc_like_current_add_params(syn_gen_nexus, syn_gen_nexus_params)

# ============== Simulation =================
stims = {'syn_gen_nexus': syn_gen_nexus}
var = L5PC_define_vars(stims=stims)
var = L5PC_record(var, L5PC, stims=stims)

modulation_params = cfg_fig_2_5["modulation"]
apic_secs = L5PC.apic
dic_gbars = nf.get_gca_hva_from_multi_sections(apic_secs)

# update apical dendrites with ACh multiplier
dic_log = {}
gCa_HVAbar_nexus_list = []
n_sp_list = []
timing_1st_list = []

for gCa_HVA_mul in modulation_params['gCa_HVA_muls']:
    tic = time.time()

    # add ACh to each segment
    for i, sec in enumerate(apic_secs):
        for j, seg in enumerate(sec):
            if hasattr(seg, 'Ca_HVA'):
                seg.Ca_HVA.gCa_HVAbar = dic_gbars[(i, j)] * gCa_HVA_mul

    run_nrn_simulation()
    simulation_data = uf.convert_var_to_numpy(var)

    gCa_HVAbar_nexus = L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_HVA.gCa_HVAbar
    gCa_HVAbar_nexus_list.append(gCa_HVAbar_nexus)

    dic_log[gCa_HVAbar_nexus] = simulation_data

    toc = time.time()
    print('finished gCa_HVA_mul %f in %f seconds' % (gCa_HVA_mul, toc - tic))


# ============ get spike properties ===========
props = {}

for gCa in gCa_HVAbar_nexus_list:
    props_curr = {}
    t = dic_log[gCa]['t']
    v_soma = dic_log[gCa]['soma']['v']
    v_nexus = dic_log[gCa]['nexus']['v']

    (log, start_t, stop_t) = af.is_calcium_spike(t, v_nexus)
    props_curr['log'] = log

    if log:
        ca_spike_dur = stop_t - start_t

        n_sp = af.n_spikes(v_soma)
        ind_peak = np.argmax(v_nexus)

        props_curr['duration'] = ca_spike_dur
        props_curr['n_sp'] = n_sp
        props_curr['start_t'] = start_t
        props_curr['stop_t'] = stop_t
        props_curr['ind_peak'] = ind_peak

    props[gCa] = props_curr


# ============ plot ==============
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm (np.linspace(0.65, 0.95, 256))

coolwarm_half = ListedColormap(newcolors)

evenly_spaced_interval = np.linspace(0, 1, len(gCa_HVAbar_nexus_list))
colors = [coolwarm_half(k) for k in evenly_spaced_interval]

lbox = 35
rbox = 50
ubox = -20
dbox = -75

fig, ax = plt.subplots(figsize=(4, 2.5))
for i, gCa in enumerate(gCa_HVAbar_nexus_list):
    t = dic_log[gCa]["t"]
    t_shifted = t - syn_gen_nexus.onset
    ax.plot(t_shifted, dic_log[gCa]["nexus"]['v'], color=colors[i])

plt.plot([lbox, lbox], [dbox, ubox], '--k')
plt.plot([rbox, rbox], [dbox, ubox], '--k')
plt.plot([lbox, rbox], [dbox, dbox], '--k')
plt.plot([lbox, rbox], [ubox, ubox], '--k')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('ms', fontsize=16)
ax.set_ylabel('mV', fontsize=16)
ax.set_xlim([-5, 55])
ax.set_ylim([-80, 50])
ax.tick_params(axis='both', labelsize=12)
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig2/fig2_add_ACh.svg", format='svg')


# =========== quantify ==========
list_dur = []
for gCa in gCa_HVAbar_nexus_list:
    dur = props[gCa]['duration']
    list_dur.append(dur)

fig, ax = plt.subplots(figsize=(3.5, 1))
for i, gCa_HVA_mul in enumerate(modulation_params["gCa_HVA_muls"]):
    ax.plot(gCa_HVA_mul, list_dur[i], 'o', color=colors[i], markersize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('mS/cm^2', fontsize=16)
ax.set_ylabel('ms', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
plt.show()

if general_params["save_figs"]:
    fig.savefig("./Fig2/fig2_add_ACh_quantify.svg", format='svg')
