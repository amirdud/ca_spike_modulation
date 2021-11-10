
import time
import numpy as np
from neuron import h,gui
import matplotlib.pyplot as plt
import matplotlib
import funs
import pickle
import seaborn as sns
import scipy.signal as ss
matplotlib.use('Qt5Agg')

##=================== creating cell object ===========================

h.load_file("import3d.hoc")
morphology_file = "morphologies/cell1.asc"
h.load_file("models/L5PCbiophys5b.hoc")
h.load_file("models/L5PCtemplate_2.hoc")

h.define_shape()
L5PC = h.L5PCtemplate(morphology_file)


## =========== Variables =================
t = h.Vector()
soma_v = h.Vector()
nexus_v = h.Vector()
stim_i = h.Vector()
stim_pert_nexus = h.Vector()
gHVA = h.Vector()
gLVA = h.Vector()
gSK_E2 = h.Vector()
gSKv3_1 = h.Vector()
gIm = h.Vector()
gIh = h.Vector()
gNaTs2_t = h.Vector()
gCaDynamics_E2 = h.Vector()
HVA_ca = h.Vector()
LVA_ca = h.Vector()
SK_k = h.Vector()
SKv3_k = h.Vector()
Im_k = h.Vector()
Ih_cn = h.Vector()
NaTs2_t_na = h.Vector()
dend_ik = h.Vector()
dend_ina = h.Vector()
dend_ica = h.Vector()

t.record(h._ref_t)
soma_v.record(L5PC.soma[0](0.5)._ref_v)
nexus_v.record(L5PC.apic[36](0.9)._ref_v)
gHVA.record(L5PC.apic[36](0.9).Ca_HVA._ref_gCa_HVA)
gLVA.record(L5PC.apic[36](0.9).Ca_LVAst._ref_gCa_LVAst)
gSK_E2.record(L5PC.apic[36](0.9).SK_E2._ref_gSK_E2)
gSKv3_1.record(L5PC.apic[36](0.9).SKv3_1._ref_gSKv3_1)
gIm.record(L5PC.apic[36](0.9).Im._ref_gIm)
gIh.record(L5PC.apic[36](0.9).Ih._ref_gIh)
gNaTs2_t.record(L5PC.apic[36](0.9).NaTs2_t._ref_gNaTs2_t)
gCaDynamics_E2.record(L5PC.apic[36](0.9).CaDynamics_E2._ref_decay)
HVA_ca.record(L5PC.apic[36](0.9).Ca_HVA._ref_ica)
LVA_ca.record(L5PC.apic[36](0.9).Ca_LVAst._ref_ica)
SK_k.record(L5PC.apic[36](0.9).SK_E2._ref_ik)
SKv3_k.record(L5PC.apic[36](0.9).SKv3_1._ref_ik)
Im_k.record(L5PC.apic[36](0.9).Im._ref_ik)
Ih_cn.record(L5PC.apic[36](0.9).Ih._ref_ihcn)
NaTs2_t_na.record(L5PC.apic[36](0.9).NaTs2_t._ref_ina)

dend_ik.record(L5PC.apic[36](0.9).k_ion._ref_ik)
dend_ica.record(L5PC.apic[36](0.9).ca_ion._ref_ica)
dend_ina.record(L5PC.apic[36](0.9).na_ion._ref_ina)

## ============== Parameters =================

h.tstop = 250
h.v_init = -80
h.celsius = 37
th = -70  # mV
y_0 = -75  # mV
th_diff_ca_spike = 0.15

# Generation parameters:
stim = h.epsp(L5PC.apic[36](0.9))

stim.tau0 = 0.5
stim.tau1 = 5
stim.onset = 150
stim.imax = 1.6
stim_i.record(stim._ref_i)

## =========== Perturbation =================
# Perturbation parameters:
tau0s = np.array([0.5])
tau1s = np.array([5])
onsets = np.round(np.arange(5, 25 + 0.5, 0.5), 2)
imaxs = np.round(np.arange(-1, 1 + 0.05, 0.05), 2)

# Perturbation sections:
apic_secs = L5PC.apic

# make framework:
areas_list = []
dic_L = {}
dic_stims_pert = {}
# stims_I_list = []
for i,sec in enumerate(apic_secs):
    dic_L[i] = sec.L

    for j,seg in enumerate(sec):
        # add synapses
        stim_pert = h.epsp(seg)

        stim_pert.tau0 = 0
        stim_pert.tau1 = 0
        stim_pert.onset = 0
        stim_pert.imax = 0

        dic_stims_pert[(i, j)] = stim_pert

        # record current in nexus
        if sec == L5PC.apic[36] and seg == L5PC.apic[36](0.884615):
            stim_pert_nexus.record(stim_pert._ref_i)
            print('found nexus!')

        # calc area
        A = seg.area()
        areas_list.append(A)

areas_np = np.array(areas_list)
areas_sum = areas_np.sum()

#%% ============== Stimulation =================
# change perturbation in every run of the simulation
dic_log = {}
# dic_log['apic_secs'] = apic_secs
dic_log['areas_sum'] = areas_sum

for imax in imaxs:
    for tau0 in tau0s:
        for tau1 in tau1s:
            for onset in onsets:
                tic = time.time()
                data = {}

                # define amp on each segment
                for i,sec in enumerate(apic_secs):
                    for j,seg in enumerate(sec):
                        A = seg.area()
                        curr_imax = imax*A/areas_sum
                        dic_stims_pert[(i,j)].tau0 = tau0
                        dic_stims_pert[(i, j)].tau1 = tau1
                        dic_stims_pert[(i,j)].imax = curr_imax
                        dic_stims_pert[(i,j)].onset = onset + stim.onset

                h.run()
                soma_v_np = np.array(soma_v)
                nexus_v_np = np.array(nexus_v)
                gHVA_np = np.array(gHVA)
                gLVA_np = np.array(gLVA)
                gSK_E2_np = np.array(gSK_E2)
                gSKv3_1_np = np.array(gSKv3_1)
                gIm_np = np.array(gIm)
                gIh_np = np.array(gIh)
                gNaTs2_t_np = np.array(gNaTs2_t)
                gCaDynamics_E2_np = np.array(gCaDynamics_E2)
                HVA_ca_np = np.array(HVA_ca)
                LVA_ca_np = np.array(LVA_ca)
                SK_k_np = np.array(SK_k)
                SKv3_k_np = np.array(SKv3_k)
                Im_k_np = np.array(Im_k)
                Ih_cn_np = np.array(Ih_cn)
                NaTs2_t_na_np = np.array(NaTs2_t_na)
                dend_ik_np = np.array(dend_ik)
                dend_ica_np = np.array(dend_ica)
                dend_ina_np = np.array(dend_ina)

                data['soma_v'] = soma_v_np
                data['nexus_v'] = nexus_v_np
                data['gHVA'] = gHVA_np
                data['gLVA'] = gLVA_np
                data['gSK_E2'] = gSK_E2_np
                data['gSKv3_1'] = gSKv3_1_np
                data['gIm'] = gIm_np
                data['gIh'] = gIh_np
                data['gNaTs2_t'] = gNaTs2_t_np
                data['gCaDynamics_E2'] = gCaDynamics_E2_np
                data['HVA_ca'] = HVA_ca_np
                data['LVA_ca'] = LVA_ca_np
                data['SK_k'] = SK_k_np
                data['SKv3_k'] = SKv3_k_np
                data['Im_k'] = Im_k_np
                data['Ih_cn'] = Ih_cn_np
                data['NaTs2_t_na'] = NaTs2_t_na_np
                data['dend_ik'] = dend_ik_np
                data['dend_ica'] = dend_ica_np
                data['dend_ina'] = dend_ina_np

                dic_log[(imax, tau0, tau1, onset)] = data

                toc = time.time()
                print('finished imax %f, tau0 %f, tau1 %f and onset %f in %f seconds'
                      % (imax, tau0, tau1, onset, toc - tic))


#%%  ======= Load ========
load_data = True

# filename = 'dic_log_Ca_Spike_EPSP_perturbation_exc_and_weak_inh_peak_and_time.pickle'
# pickle.dump(dic_log, open(filename, 'wb'), protocol=2)

# load file
if load_data:
    filename = 'dic_log_Ca_Spike_EPSP_perturbation_exc_and_weak_inh_peak_and_time.pickle'
    file = open(filename,'rb')
    dic_log = pickle.load(file)
    file.close()

t_np = np.arange(0, h.tstop + 0.025, 0.025)
t_shifted_np = t_np - stim.onset

#%%
# ============ get spike properties ===========
props = {}
for imax in imaxs:
    for tau0 in tau0s:
        for tau1 in tau1s:
            for onset in onsets:
                props_curr = {}
                soma_v_np = dic_log[(imax, tau0, tau1, onset)]['soma_v']
                nexus_v_np = dic_log[(imax, tau0, tau1, onset)]['nexus_v']

                (log, start_t, stop_t) = funs.is_calcium_spike(t_np, nexus_v_np)
                props_curr['log'] = log

                if log:
                    ca_spike_dur = stop_t - start_t

                    [dummy, start_ind] = funs.find_nearest(t_np, start_t)
                    [dummy, stop_ind] = funs.find_nearest(t_np, stop_t)
                    area = funs.calc_integral(t_np, nexus_v_np, start_ind, stop_ind, y_0)
                    n_sp = funs.n_spikes(soma_v_np)

                    props_curr['area'] = area
                    props_curr['duration'] = ca_spike_dur
                    props_curr['n_sp'] = n_sp
                    props_curr['start_t'] = start_t
                    props_curr['stop_t'] = stop_t

                else:
                    ca_spike_dur = 0

                props[(imax, tau0, tau1, onset)] = props_curr


#%%
# ========= duration ==========

dur_matrix = np.zeros((imaxs.size, onsets.size))
for i,imax in enumerate(imaxs):
    for j, onset in enumerate(onsets):
        if props[(imax, tau0, tau1, onset)]['log'] == 1:
            dur_matrix[i, j] = props[(imax, tau0, tau1, onset)]['duration']

fig,ax = plt.subplots(figsize=(2,2))
ax = sns.heatmap(dur_matrix, cmap='gray', vmin=34, vmax=40)
ax.invert_yaxis()
ax.set_ylabel('imax')
ax.set_xlabel('onset')
ax.set_title('Spike Duration')
plt.show()

