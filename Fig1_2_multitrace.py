import time
from neuron import h, gui
import matplotlib.pyplot as plt
import pickle
import analysis_funs as af
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from cfg import *

general_params = cfg_fig_1_2["h"]
nf.define_nrn_general_params(general_params)

# get model
L5PC = nf.get_L5PC_model()

# soma current injection
soma = nf.get_soma(L5PC)
stim_soma_params = cfg_fig_1_2["stim_soma"]
stim_soma = nf.iclamp_current_insert(soma)

# nexus current injection
nexus = nf.get_nexus(L5PC)
syn_nexus_params = cfg_fig_1_2["syn_nexus"]
syn_nexus = nf.epsc_like_current_insert(nexus)

# run simulation
stims = {'syn_nexus': syn_nexus, 'stim_soma': stim_soma}
var = L5PC_define_vars(stims=stims)
var = L5PC_record(var, L5PC, stims=stims)

## ============== simulation =================

if general_params['run_sim']:
    print('Running Simulation...')
    dic_log = {}
    for delay in stim_soma_params['delay']:
        for amp in stim_soma_params['amp']:
            for dur in stim_soma_params['dur']:
                for tau0 in syn_nexus_params['tau0']:
                    for tau1 in syn_nexus_params['tau1']:
                        for BACdt in syn_nexus_params['BACdt']:
                            for imax in syn_nexus_params['imax']:
                                tic = time.time()

                                # stim params
                                stim_soma_params_curr = {'delay': delay,
                                                   'amp': amp,
                                                   'dur': dur}
                                nf.iclamp_current_add_params(stim_soma, stim_soma_params_curr)

                                # syn params
                                sym_nexus_params_curr = {'tau0': tau0,
                                                         'tau1': tau1,
                                                         'onset': stim_soma_params_curr['delay'] + dur + BACdt,
                                                         'imax': imax}
                                nf.epsc_like_current_add_params(syn_nexus, sym_nexus_params_curr)

                                run_nrn_simulation()
                                simulation_data = uf.convert_var_to_numpy(var)

                                dic_log[(amp, dur, tau0, tau1, BACdt, imax)] = simulation_data
                                toc = time.time()

                                print(f'finished: amp = {amp:.2f}, dur = {dur:.2f}, '
                                      f'tau0 = {tau0:.2f}, tau1 = {tau1:.2f}, '
                                      f'BACdt = {BACdt:.2f}, imax = {imax:.2f} '
                                      f'in {toc - tic:.2f} sec')

    # save
    print('Saving...')
    filename = 'dic_log_L5PC_traces_fig1.pickle'
    pickle.dump(dic_log, open(filename, 'wb'), protocol=2)

else:
    # load
    print('Loading...')
    filename = 'dic_log_L5PC_traces_fig1.pickle'
    file = open(filename, 'rb')
    dic_log = pickle.load(file)
    file.close()

# %%
## ========= get spike paramters ========

props = {}
for delay in stim_soma_params['delay']:
    for amp in stim_soma_params['amp']:
        for dur in stim_soma_params['dur']:
            for tau0 in syn_nexus_params['tau0']:
                for tau1 in syn_nexus_params['tau1']:
                    for BACdt in syn_nexus_params['BACdt']:
                        for imax in syn_nexus_params['imax']:
                            props_curr = {}
                            t = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['t']
                            v_soma = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['soma']['v']
                            v_nexus = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['nexus']['v']

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

                            props[(amp, dur, tau0, tau1, BACdt, imax)] = props_curr


# =========== plot =============

fig, ax = plt.subplots()
for delay in stim_soma_params['delay']:
    for amp in stim_soma_params['amp']:
        for dur in stim_soma_params['dur']:
            for tau0 in syn_nexus_params['tau0']:
                for tau1 in syn_nexus_params['tau1']:
                    for BACdt in syn_nexus_params['BACdt']:
                        for imax in syn_nexus_params['imax']:
                            t = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['t']
                            v_nexus = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['nexus']['v']
                            ax.plot(t, v_nexus, 'k', label='nexus_v')
                            ax.set_ylabel('mV')
                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)

plt.show()

# %%
## =========== barplot =============
bef_peak_ind = int(general_params["bef_peak_t"] / h.dt)
aft_peak_ind = int(general_params["aft_peak_t"] / h.dt)
ind4 = int(general_params["t4"] / h.dt)
ind3 = int(general_params["t3"] / h.dt)
ind2 = int(general_params["t2"] / h.dt)
ind1 = int(general_params["t1"] / h.dt)

dend_v_shifted_all = []
m_dvdt_dend_v_P1 = []
m_dvdt_dend_v_P2 = []
m_dvdt_dend_v_P3 = []

dvdt_dend_v_P1 = []
dvdt_dend_v_P2 = []
dvdt_dend_v_P3 = []

for delay in stim_soma_params['delay']:
    for amp in stim_soma_params['amp']:
        for dur in stim_soma_params['dur']:
            for tau0 in syn_nexus_params['tau0']:
                for tau1 in syn_nexus_params['tau1']:
                    for BACdt in syn_nexus_params['BACdt']:
                        for imax in syn_nexus_params['imax']:
                            try:
                                ind_peak = props[(amp, dur, tau0, tau1, BACdt, imax)]['ind_peak']
                            except:
                                continue
                            t = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['t']
                            t_shifted = t[ind_peak - bef_peak_ind:ind_peak + aft_peak_ind] - t[ind_peak:][0]

                            v_soma = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['soma']['v']
                            v_nexus = dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['nexus']['v']

                            v_nexus_shifted = v_nexus[ind_peak - bef_peak_ind:ind_peak + aft_peak_ind]
                            v_soma_shifted = v_soma[ind_peak - bef_peak_ind:ind_peak + aft_peak_ind]

                            v_nexus_shifted_dvdt = np.diff(v_nexus_shifted)
                            dvdt_dend_v_P1_i = np.diff(v_nexus[ind_peak - ind2:ind_peak - ind1]).reshape((1, -1))
                            dvdt_dend_v_P2_i = np.diff(v_nexus[ind_peak + ind1:ind_peak + ind2]).reshape((1, -1))
                            dvdt_dend_v_P3_i = np.diff(v_nexus[ind_peak + ind3:ind_peak + ind4]).reshape((1, -1))

                            dvdt_dend_v_P1.append(dvdt_dend_v_P1_i)
                            dvdt_dend_v_P2.append(dvdt_dend_v_P2_i)
                            dvdt_dend_v_P3.append(dvdt_dend_v_P3_i)

dvdt_dend_v_P1_np = np.concatenate(dvdt_dend_v_P1, axis=0)
dvdt_dend_v_P2_np = np.concatenate(dvdt_dend_v_P2, axis=0)
dvdt_dend_v_P3_np = np.concatenate(dvdt_dend_v_P3, axis=0)

P1_std = np.std(dvdt_dend_v_P1_np, axis=0)
P2_std = np.std(dvdt_dend_v_P2_np, axis=0)
P3_std = np.std(dvdt_dend_v_P3_np, axis=0)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.plot(P1_std)
ax1.plot(P2_std)
ax2.plot(P3_std)
plt.show()

# %% barplot
m_dvdt_dend_v = [P1_std, P2_std, P3_std]
xs = [np.ones(P1_std.shape), 2 * np.ones(P1_std.shape), 3 * np.ones(P3_std.shape)]
ave_xs = [1, 2, 3]
ave_m_dvdt_dend_v = [P1_std.mean(), P2_std.mean(), P3_std.mean()]
std_m_dvdt_dend_v = [P1_std.std(), P2_std.std(), P3_std.std()]

fig, ax = plt.subplots(figsize=(1.2, 1.5))
medianprops = {'color': 'w'}
colors = ['k', 'k', 'k']
barp = ax.bar(ave_xs, ave_m_dvdt_dend_v, yerr=std_m_dvdt_dend_v, color='k')

plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig1/fig1_boxplot.svg", format='svg')
