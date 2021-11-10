
import numpy as np
from neuron import h,gui
import matplotlib.pyplot as plt
import matplotlib
import funs
import time
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
dend_v = h.Vector()
stim_nexus = h.Vector()

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
soma_ik = h.Vector()
soma_ina = h.Vector()
soma_ica = h.Vector()

t.record(h._ref_t)
soma_v.record(L5PC.soma[0](0.5)._ref_v)
dend_v.record(L5PC.apic[36](0.9)._ref_v)
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
soma_ik.record(L5PC.soma[0](0.5).k_ion._ref_ik)
soma_ica.record(L5PC.soma[0](0.5).ca_ion._ref_ica)
soma_ina.record(L5PC.soma[0](0.5).na_ion._ref_ina)

## ============== Parameters =================
h.dt = 0.025
h.tstop = 250
h.v_init = -80
h.celsius = 37
y_0 = -42

amps = np.round(np.arange(0.8, 2.6 + 0.2, 0.2), 2)

# make framework:
all_secs = L5PC.all

areas_list = []
dic_L = {}
dic_stims = {}
for i,sec in enumerate(all_secs):
    dic_L[i] = sec.L

    for j,seg in enumerate(sec):
        # add iclamp
        stim =  h.IClamp(seg)
        stim.dur = 1000
        stim.delay = 0

        dic_stims[(i, j)] = stim

        # record current in nexus
        if sec == L5PC.apic[36] and seg == L5PC.apic[36](0.884615):
            stim_nexus.record(stim._ref_i)
            print('found nexus!')

        # calc area
        A = seg.area()
        areas_list.append(A)

areas_np = np.array(areas_list)
areas_sum = areas_np.sum()

## ============== run simulation ===============
# change constant current in every run of the simulation
dic_log = {}
dic_log['areas_sum'] = areas_sum

for amp in amps:
    tic = time.time()

    data = {}

    # define amp on each segment
    for i, sec in enumerate(all_secs):
        for j, seg in enumerate(sec):
            A = seg.area()
            curr_amp = amp * A / areas_sum
            dic_stims[(i, j)].amp = curr_amp

    h.run()
    soma_v_np = np.array(soma_v)
    dend_v_np = np.array(dend_v)
    stim_nexus_np = np.array(stim_nexus)
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
    soma_ik_np = np.array(soma_ik)
    soma_ica_np = np.array(soma_ica)
    soma_ina_np = np.array(soma_ina)

    data['soma_v'] = soma_v_np
    data['dend_v'] = dend_v_np

    dic_log[amp] = data

    toc = time.time()
    print('finished amp %f in %f seconds'% (amp,toc-tic))

t_np = np.arange(0, h.tstop + 0.025, 0.025)

#%%
# ============ get spike properties ===========
props = {}
for amp in amps:
    props_curr = {}
    soma_v_np = dic_log[amp]['soma_v']
    nexus_v_np = dic_log[amp]['dend_v']
    t_np = np.arange(0, h.tstop + 0.025, 0.025)

    (log, start_t, stop_t) = funs.is_calcium_spike(t_np, nexus_v_np)
    props_curr['log'] = log

    if log:
        ca_spike_dur = stop_t - start_t

        [dummy, start_ind] = funs.find_nearest(t_np, start_t)
        [dummy, stop_ind] = funs.find_nearest(t_np, stop_t)
        area = funs.calc_integral(t_np, nexus_v_np, start_ind, stop_ind, y_0)

        # max plateau value in specific t
        t_plat = 36.8 / 2  # middle point in the first plateau in the loop
        [dummy, ind_plat] = funs.find_nearest(t_np[start_ind:] - t_np[start_ind:][0], t_plat)  # the ca spike moves so we align it and take a value in the middle of the first spike
        max_plat = nexus_v_np[start_ind:][ind_plat]

        # ind of Ca spike peak
        ind_peak = np.argmax(nexus_v_np)

        props_curr['area'] = area
        props_curr['duration'] = ca_spike_dur
        props_curr['start_t'] = start_t
        props_curr['stop_t'] = stop_t
        props_curr['max_plat'] = max_plat
        props_curr['ind_peak'] = ind_peak

    else:
        ca_spike_dur = 0

    props[amp] = props_curr


#%%
# ============ plot voltages =============
# align ca spikes to check duration
from matplotlib import cm
from matplotlib.colors import ListedColormap

coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0.65, 0.95, 256))

coolwarm_half = ListedColormap(newcolors)

evenly_spaced_interval = np.linspace(0, 1, len(amps))
colors = [coolwarm_half(k) for k in evenly_spaced_interval]

lbox = 25
rbox = 40
ubox = -25
dbox = -70

bef_peak_t = 5 # msec
bef_peak_ind = int(bef_peak_t/h.dt)


fig,ax = plt.subplots(figsize=(4,2.5))
for i,amp in enumerate(amps):
    ind_peak = props[amp]['ind_peak']

    # ind_start = funs.find_nearest(t_np,t_start)[1]
    plt.plot(t_np[ind_peak-bef_peak_ind:]-t_np[ind_peak:][0],dic_log[amp]['dend_v'][ind_peak-bef_peak_ind:],linewidth=2,color=colors[i],label='nexus')

    plt.plot([lbox, lbox], [dbox, ubox], '--k')
    plt.plot([rbox, rbox], [dbox, ubox], '--k')
    plt.plot([lbox, rbox], [dbox, dbox], '--k')
    plt.plot([lbox, rbox], [ubox, ubox], '--k')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('ms', fontsize=16)
ax.set_ylabel('mV', fontsize=16)
ax.set_xlim([-bef_peak_t, 55])
ax.set_ylim([-80, 50])
ax.tick_params(axis='both', labelsize=12)

plt.show()
# fig.savefig("./Fig2/fig2_iclamp_const_R2.svg", format='svg')


# ============ quantify =============

fig, ax = plt.subplots(figsize=(3.5, 1))
for i, amp in enumerate(amps):
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
# fig.savefig("./Fig2/fig2_iclamp_const_quantify.svg", format='svg')
