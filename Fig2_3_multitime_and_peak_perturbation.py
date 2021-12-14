import time
import pickle
from neuron import h,gui
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_funs as af
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from cfg import *


general_params = cfg_fig_2_3["h"]
nf.define_nrn_general_params(general_params)

# ======= model =======
L5PC = nf.get_L5PC_model()

# ========== generation ==========
nexus = nf.get_nexus(L5PC)
syn_gen_nexus_params = cfg_fig_2_3["syn_gen_nexus"]
syn_gen_nexus = nf.epsc_like_current_insert(nexus)
nf.epsc_like_current_add_params(syn_gen_nexus, syn_gen_nexus_params)

# ========== perturbation ==========
syn_pert_nexus_params = cfg_fig_2_3["syn_pert_nexus"]
apic_secs = L5PC.apic
output_dic = nf.epsc_like_current_insert_to_multi_sections(L5PC, apic_secs)

# ============== Simulation =================
dic_syns_pert = output_dic["dic_syns_pert"]
nexus_inds = output_dic["nexus_inds"]
areas_sum = output_dic["areas_sum"]

syn_pert_nexus = dic_syns_pert[nexus_inds]
stims = {'syn_gen_nexus': syn_gen_nexus, 'syn_pert_nexus': syn_pert_nexus}
var = L5PC_define_vars(stims=stims)
var = L5PC_record(var, L5PC, stims=stims)

# change perturbation in every run of the simulation
if general_params['run_sim']:
    print('Running Simulation...')
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
    # save
    print('Saving...')
    filename = 'dic_log_Ca_Spike_EPSP_perturbation_exc_and_weak_inh_peak_and_time.pickle'
    pickle.dump(dic_log, open(filename, 'wb'), protocol=2)

else:
    # load
    print('Loading...')
    filename = 'dic_log_Ca_Spike_EPSP_perturbation_exc_and_weak_inh_peak_and_time.pickle'
    file = open(filename, 'rb')
    dic_log = pickle.load(file)
    file.close()

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
dur_matrix = np.zeros((syn_pert_nexus_params['imax'].size, syn_pert_nexus_params['onset'].size))
for i,imax in enumerate(syn_pert_nexus_params['imax']):
    for j, onset in enumerate(syn_pert_nexus_params['onset']):
        if props[(imax, tau0, tau1, onset)]['log'] == 1:
            dur_matrix[i, j] = props[(imax, tau0, tau1, onset)]['duration']

fig, ax = plt.subplots(figsize=(2, 2))
ax = sns.heatmap(dur_matrix, cmap='gray', vmin=34, vmax=40)
ax.invert_yaxis()
ax.set_ylabel('imax')
ax.set_xlabel('onset')
ax.set_title('Spike Duration')
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig2/fig2_time_and_peak_heatmap.svg", format='svg')
