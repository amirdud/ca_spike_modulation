import time
from neuron import h,gui
import matplotlib.pyplot as plt
from matplotlib import cm
import analysis_funs as af
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from cfg import *

general_params = cfg_fig_2_1["h"]
nf.define_nrn_general_params(general_params)

# ======= model =======
L5PC = nf.get_L5PC_model()

# ========== generation ==========
nexus = nf.get_nexus(L5PC)
syn_gen_nexus_params = cfg_fig_2_1["syn_gen_nexus"]
syn_gen_nexus = nf.epsc_like_current_insert(nexus)
nf.epsc_like_current_add_params(syn_gen_nexus, syn_gen_nexus_params)

# ========== perturbation ==========
syn_pert_nexus_params = cfg_fig_2_1["syn_pert_nexus"]
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

for imax in syn_pert_nexus_params["imax"]:
    for tau0 in syn_pert_nexus_params["tau0"]:
        for tau1 in syn_pert_nexus_params["tau1"]:
            for onset in syn_pert_nexus_params["onset"]:
                tic = time.time()

                # define amp on each segment
                for i, sec in enumerate(apic_secs):
                    for j, seg in enumerate(sec):
                        A = seg.area()
                        curr_imax = imax * A / areas_sum
                        dic_syns_pert[(i, j)].tau0 = tau0
                        dic_syns_pert[(i, j)].tau1 = tau1
                        dic_syns_pert[(i, j)].imax = curr_imax
                        dic_syns_pert[(i, j)].onset = onset + syn_gen_nexus.onset

                run_nrn_simulation()
                simulation_data = uf.convert_var_to_numpy(var)
                dic_log[(imax, tau0, tau1, onset)] = simulation_data

                toc = time.time()
                print('finished imax %f, tau0 %f, tau1 %f and onset %f in %f seconds'
                      % (imax, tau0, tau1, onset, toc - tic))


# ============ get spike properties ===========
props = {}
for imax in syn_pert_nexus_params["imax"]:
    for tau0 in syn_pert_nexus_params["tau0"]:
        for tau1 in syn_pert_nexus_params["tau1"]:
            for onset in syn_pert_nexus_params["onset"]:
                props_curr = {}
                t = dic_log[(imax, tau0, tau1, onset)]['t']
                v_soma = dic_log[(imax, tau0, tau1, onset)]['soma']['v']
                v_nexus = dic_log[(imax, tau0, tau1, onset)]['nexus']['v']

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

                props[(imax, tau0, tau1, onset)] = props_curr


# ============ plot ==============
evenly_spaced_interval = np.linspace(0, 1, len(syn_pert_nexus_params["imax"]))
colors = [cm.coolwarm(k) for k in evenly_spaced_interval]

lbox = 30
rbox = 50
ubox = -20
dbox = -75

fig, ax = plt.subplots(figsize=(4, 2.5))
for i, imax in enumerate(syn_pert_nexus_params["imax"]):
    for tau0 in syn_pert_nexus_params["tau0"]:
        for tau1 in syn_pert_nexus_params["tau1"]:
            for onset in syn_pert_nexus_params["onset"]:
                t_shifted = dic_log[(imax, tau0, tau1, onset)]["t"] - syn_gen_nexus.onset

                if imax == 0:
                    c = 'k'
                else:
                    c = colors[i]
                plt.plot(t_shifted, dic_log[(imax, tau0, tau1, onset)]['nexus']['v'], color=c)

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
    fig.savefig("./Fig2/fig2_EPSP_exc_inh.svg", format='svg')


# =========== quantify ==========
fig, ax = plt.subplots(figsize=(3.5, 1))
for i, imax in enumerate(syn_pert_nexus_params["imax"]):
    for tau0 in syn_pert_nexus_params["tau0"]:
        for tau1 in syn_pert_nexus_params["tau1"]:
            for onset in syn_pert_nexus_params["onset"]:
                dur = props[(imax, tau0, tau1, onset)]['duration']
                plt.plot(imax, dur, 'o', color=colors[i],markersize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('perturbation (nA)', fontsize=16)
ax.set_ylabel('ms', fontsize=16)

plot_condition = general_params["plot_condition"]
if plot_condition == 'strong_inh':
    ax.set_xlim([-6.25, -2.75])
    ax.set_ylim([27, 35.25])
else:
    ax.set_xlim([-2.75, 6.25])
    ax.set_ylim([35.25, 39.25])
ax.tick_params(axis='both', labelsize=12)

plt.show()
if general_params["save_figs"]:
    if plot_condition == 'strong_inh':
        fig.savefig("./Fig2/fig2_EPSP_strong_inh_quantify.svg", format='svg')
    else:
        fig.savefig("./Fig2/fig2_EPSP_exc_weak_inh_quantify.svg", format='svg')
