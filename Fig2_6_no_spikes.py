
import time
from neuron import h,gui
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
import analysis_funs as af
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from cfg import *


general_params = cfg_fig_2_6["h"]
nf.define_nrn_general_params(general_params)

# ======= model =======
L5PC = nf.get_L5PC_model()

# ========== generation ==========
nexus = nf.get_nexus(L5PC)
syn_gen_nexus_params = cfg_fig_2_6["syn_gen_nexus"]
syn_gen_nexus = nf.epsc_like_current_insert(nexus)
nf.epsc_like_current_add_params(syn_gen_nexus, syn_gen_nexus_params)

## =========== delete spikes in the soma and axon =================

soma_sections = L5PC.soma
axon_sections = L5PC.axon

factor = general_params["factor"]
for sec_soma_i in soma_sections:
    for seg_soma_i in sec_soma_i:
        seg_soma_i.NaTs2_t.gNaTs2_tbar = seg_soma_i.NaTs2_t.gNaTs2_tbar/factor
        seg_soma_i.Ca_LVAst.gCa_LVAstbar = seg_soma_i.Ca_LVAst.gCa_LVAstbar/factor
        seg_soma_i.Ca_HVA.gCa_HVAbar = seg_soma_i.Ca_HVA.gCa_HVAbar/factor

for sec_axon_i in axon_sections:
    for seg_axon_i in sec_axon_i:
        seg_axon_i.NaTa_t.gNaTa_tbar = seg_axon_i.NaTa_t.gNaTa_tbar/factor
        seg_axon_i.Nap_Et2.gNap_Et2bar = seg_axon_i.Nap_Et2.gNap_Et2bar/factor
        seg_axon_i.Ca_LVAst.gCa_LVAstbar = seg_axon_i.Ca_LVAst.gCa_LVAstbar/factor
        seg_axon_i.Ca_HVA.gCa_HVAbar = seg_axon_i.Ca_HVA.gCa_HVAbar/factor

# ========== perturbation ==========
syn_pert_nexus_params = cfg_fig_2_6["syn_pert_nexus"]

imaxs_strong_ipsp = syn_pert_nexus_params["imax_strong_ipsp"]
imaxs_weak_ipsp = syn_pert_nexus_params["imax_weak_ipsp"]
imaxs_epsp = syn_pert_nexus_params["imax_epsp"]
imaxs = np.concatenate((imaxs_strong_ipsp, imaxs_weak_ipsp, imaxs_epsp))
imaxs_type_list = [imaxs_strong_ipsp, imaxs_weak_ipsp, imaxs_epsp]  # for color

apic_secs = L5PC.apic
output_dic = nf.epsc_like_current_insert_to_multi_sections(L5PC, apic_secs)

# ============== simulation =================
dic_syns_pert = output_dic["dic_syns_pert"]
nexus_inds = output_dic["nexus_inds"]
areas_sum = output_dic["areas_sum"]

syn_pert_nexus = dic_syns_pert[nexus_inds]
stims = {'syn_gen_nexus': syn_gen_nexus, 'syn_pert_nexus': syn_pert_nexus}
var = L5PC_define_vars(stims=stims)
var = L5PC_record(var, L5PC, stims=stims)

# change perturbation in every run of the simulation
dic_log = {}

for imax in imaxs:
    tic = time.time()

    # define amp on each segment
    for i, sec in enumerate(apic_secs):
        for j, seg in enumerate(sec):
            A = seg.area()
            curr_imax = imax * A / areas_sum
            dic_syns_pert[(i, j)].tau0 = syn_pert_nexus_params["tau0"]
            dic_syns_pert[(i, j)].tau1 = syn_pert_nexus_params["tau1"]
            dic_syns_pert[(i, j)].imax = curr_imax
            dic_syns_pert[(i, j)].onset = syn_pert_nexus_params["onset"] + syn_gen_nexus.onset

    run_nrn_simulation()
    simulation_data = uf.convert_var_to_numpy(var)
    dic_log[imax] = simulation_data

    toc = time.time()
    print('finished imax %f in %f seconds' % (imax, toc - tic))


# ============ get spike properties ===========
props = {}
for imax in imaxs:
    props_curr = {}
    t = dic_log[imax]['t']
    v_soma = dic_log[imax]['soma']['v']
    v_nexus = dic_log[imax]['nexus']['v']

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

    props[imax] = props_curr

# ============ plot ==============

coolwarms = cm.get_cmap('coolwarm', 256)
newcolors_reds = coolwarms(np.linspace(0.5, 1, 256))  # PSC: 0 - 2
newcolors_strong_blues = coolwarms(np.linspace(0, (5 - 4) / 5 * 0.5, 256))  # PSC: -5 - -4
newcolors_weak_blues = coolwarms(np.linspace((5 - 1.75) / 5 * 0.5, (5 - 0.25) / 5 * 0.5, 256))  # PSC: -1.75 - -0.25
strong_blues_part = ListedColormap(newcolors_strong_blues)
weak_blues_part = ListedColormap(newcolors_weak_blues)
reds_part = ListedColormap(newcolors_reds)
strong_blues_part_interval = np.linspace(0, 1, len(imaxs_strong_ipsp))
weak_blues_part_interval = np.linspace(0, 1, len(imaxs_weak_ipsp))
reds_part_interval = np.linspace(0, 1, len(imaxs_epsp))

colors_strong_blues = [strong_blues_part(k) for k in strong_blues_part_interval]
colors_weak_blues = [weak_blues_part(k) for k in weak_blues_part_interval]
colors_reds = [reds_part(k) for k in reds_part_interval]

lbox = 40
rbox = 50
ubox = -25
dbox = -70

fig, ax = plt.subplots(figsize=(4, 2.5))
for imax_type in imaxs_type_list:
    for i, imax in enumerate(imax_type):
        if imax == 0:
            c = 'k'
        elif imax in imaxs_strong_ipsp:
            c = colors_strong_blues[i]
        elif imax in imaxs_weak_ipsp:
            c = colors_weak_blues[i]
        elif imax in imaxs_epsp:
            c = colors_reds[i]

        t_shifted = dic_log[imax]["t"] - syn_gen_nexus.onset
        plt.plot(t_shifted, dic_log[imax]['nexus']['v'], color=c)

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
if general_params["save_figs"]:
    fig.savefig("./Fig2/Fig2_EPSP_no_somatic_spikes.svg", format='svg')


# ============ quantify ==============

fig, ax = plt.subplots(figsize=(3.5, 1))
for imax_type in imaxs_type_list:
    for i, imax in enumerate(imax_type):

        if imax in imaxs_strong_ipsp:
            c = colors_strong_blues[i]
        elif imax in imaxs_weak_ipsp:
            c = colors_weak_blues[i]
        elif imax in imaxs_epsp:
            c = colors_reds[i]

        dur = props[imax]['duration']
        plt.plot(imax, dur, 'o', color=c, markersize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('perturbation (nA)', fontsize=16)
ax.set_ylabel('ms', fontsize=16)
if general_params["plot_condition"] == 'epsp_and_weak_ipsp':
    ax.set_xlim([-2, 2])
    ax.set_ylim([38.5, 40.5])

if general_params["plot_condition"] == 'strong_ipsp':
    ax.set_xlim([-5.25, -3.75])
    ax.set_ylim([37.5, 39.0])

ax.tick_params(axis='both', labelsize=12)
plt.show()
if general_params["save_figs"]:
    if general_params["plot_condition"] == 'epsp_and_weak_ipsp':
        fig.savefig("./FigS1/figS1_EPSP_exc_weak_inh_no_somatic_spikes_quantify.svg", format='svg')
    if general_params["plot_condition"] == 'strong_ipsp':
        fig.savefig("./FigS1/figS1_EPSP_strong_inh_no_somatic_spikes_quantify.svg", format='svg')