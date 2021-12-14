import time
from neuron import h,gui
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import analysis_funs as af
import util_funs as uf
import nrn_funs as nf
from define_simulation_funs import *
from cfg import *

general_params = cfg_fig_2_4["h"]
nf.define_nrn_general_params(general_params)

# ======= model =======
L5PC = nf.get_L5PC_model()

# ========== constant current ==========
stim_constant_nexus_params = cfg_fig_2_4["stim_constant_nexus"]
all_secs = L5PC.all
output_dic = nf.constant_current_insert_to_multi_sections(L5PC, all_secs)

# ============== Simulation =================
dic_stims = output_dic["dic_stims"]
nexus_inds = output_dic["nexus_inds"]
areas_sum = output_dic["areas_sum"]

stim_constant_nexus = dic_stims[nexus_inds]
stims = {'stim_constant_nexus': stim_constant_nexus}
var = L5PC_define_vars(stims=stims)
var = L5PC_record(var, L5PC, stims=stims)

# change perturbation in every run of the simulation
dic_log = {}

for amp in stim_constant_nexus_params["amp"]:
    tic = time.time()

    # define amp on each segment
    for i, sec in enumerate(all_secs):
        for j, seg in enumerate(sec):
            A = seg.area()
            curr_amp = amp * A / areas_sum
            dic_stims[(i, j)].amp = curr_amp

    run_nrn_simulation()
    simulation_data = uf.convert_var_to_numpy(var)
    dic_log[amp] = simulation_data

    toc = time.time()
    print('finished amp %f in %f seconds' % (amp, toc - tic))


# ============ get spike properties ===========
props = {}
for amp in stim_constant_nexus_params["amp"]:
    props_curr = {}
    t = dic_log[amp]['t']
    v_soma = dic_log[amp]['soma']['v']
    v_nexus = dic_log[amp]['nexus']['v']

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

    props[amp] = props_curr

# ============ plot ==============
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0.65, 0.95, 256))

coolwarm_half = ListedColormap(newcolors)

evenly_spaced_interval = np.linspace(0, 1, len(stim_constant_nexus_params["amp"]))
colors = [coolwarm_half(k) for k in evenly_spaced_interval]

lbox = 25
rbox = 40
ubox = -25
dbox = -70

bef_peak_ind = int(general_params['bef_peak_t']/general_params['dt'])

fig, ax = plt.subplots(figsize=(4, 2.5))
for i, amp in enumerate(stim_constant_nexus_params["amp"]):
    ind_peak = props[amp]['ind_peak']
    t = dic_log[amp]["t"]
    v_nexus = dic_log[amp]['nexus']['v']

    plt.plot(t[ind_peak - bef_peak_ind:] - t[ind_peak:][0], v_nexus[ind_peak - bef_peak_ind:],
             linewidth=2, color=colors[i], label='nexus')


    plt.plot([lbox, lbox], [dbox, ubox], '--k')
    plt.plot([rbox, rbox], [dbox, ubox], '--k')
    plt.plot([lbox, rbox], [dbox, dbox], '--k')
    plt.plot([lbox, rbox], [ubox, ubox], '--k')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('ms', fontsize=16)
ax.set_ylabel('mV', fontsize=16)
ax.set_xlim([-general_params['bef_peak_t'], 55])
ax.set_ylim([-80, 50])
ax.tick_params(axis='both', labelsize=12)

plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig2/fig2_iclamp_const.svg", format='svg')


# =========== quantify ==========
fig, ax = plt.subplots(figsize=(3.5, 1))
for i, amp in enumerate(stim_constant_nexus_params["amp"]):
    dur = props[amp]['duration']
    plt.plot(amp, dur, 'o', color=colors[i], markersize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('amplitude (nA)', fontsize=16)
ax.set_ylabel('ms', fontsize=16)
ax.set_xlim([0.7, 2.7])
ax.set_ylim([30, 37])
ax.tick_params(axis='both', labelsize=12)
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig2/fig2_iclamp_const_quantify.svg", format='svg')
