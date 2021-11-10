
import numpy as np
from neuron import h,gui
import matplotlib.pyplot as plt
import matplotlib
import funs
matplotlib.use('Qt5Agg')
import time

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
stim_i = h.Vector()
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


## ============== parameters =================
h.dt = 0.025
h.tstop = 250
h.v_init = -80
h.celsius = 37
th = -70 # mV
y_0 = -75 # mV
th_diff_ca_spike = 0.15

# Generation parameters:
stim = h.epsp(L5PC.apic[36](0.9))

stim.tau0 = 0.5
stim.tau1 = 5
stim.imax = 1.6
stim.onset = 150
stim_i.record(stim._ref_i)


## ============== run simulation ===============
# add ACh every simulation
apic_secs = list(L5PC.apic)

gCa_HVA_muls = np.round(np.arange(1, 1.2 + 0.02, 0.02), 2)

# get gbars:
dic_gbars = {}
for i,sec in enumerate(apic_secs):
    for j,seg in enumerate(sec):
        dic_gbars[(i, j)] = seg.gCa_HVAbar_Ca_HVA


dic_log = {}
gCa_HVAbar_nexus_list = []
n_sp_list = []
timing_1st_list = []

# update apical dendrites with ACh value
for gCa_HVA_mul in gCa_HVA_muls:
    data = {}

    tic = time.time()

    # define amp on each segment
    for i, sec in enumerate(apic_secs):
        for j, seg in enumerate(sec):
            # check if segment has Ca_HVA
            if hasattr(seg, 'Ca_HVA'):

                seg.Ca_HVA.gCa_HVAbar = dic_gbars[(i, j)]*gCa_HVA_mul

    h.run()

    gCa_HVAbar_nexus = L5PC.apic[36](0.9).Ca_HVA.gCa_HVAbar
    gCa_HVAbar_nexus_list.append(gCa_HVAbar_nexus)

    soma_v_np = np.array(soma_v)
    dend_v_np = np.array(dend_v)
    stim_i_np = np.array(stim_i)
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

    n_sp = funs.n_spikes(soma_v_np)
    n_sp_list.append(n_sp)

    Fs =1/h.dt
    timing_1st = funs.time_first_spike(soma_v_np, Fs)
    timing_1st_list.append(timing_1st)

    data['soma_v'] = soma_v_np
    data['dend_v'] = dend_v_np

    dic_log[gCa_HVAbar_nexus] = data

    toc = time.time()
    print(toc-tic)

t_np = np.array(t)
t_np_shifted = t_np - stim.onset

#%%
# ============= plot voltages ===========
from matplotlib import cm
from matplotlib.colors import ListedColormap

coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm (np.linspace(0.65, 0.95, 256))

coolwarm_half = ListedColormap(newcolors)

evenly_spaced_interval = np.linspace(0, 1, len(gCa_HVAbar_nexus_list))
colors = [coolwarm_half(k) for k in evenly_spaced_interval]

lbox = 35
rbox = 50
ubox = -20
dbox = -75

fig,ax = plt.subplots(figsize=(4,2.5))
for i,gCa in enumerate(gCa_HVAbar_nexus_list):
    ax.plot(t_np_shifted, dic_log[gCa]['dend_v'],color=colors[i])

plt.plot([lbox, lbox], [dbox, ubox], '--k')
plt.plot([rbox, rbox], [dbox, ubox], '--k')
plt.plot([lbox, rbox], [dbox, dbox], '--k')
plt.plot([lbox, rbox], [ubox, ubox], '--k')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('ms',fontsize = 16)
ax.set_ylabel('mV',fontsize = 16)
ax.set_xlim([-5,55])
ax.set_ylim([-80,50])
ax.tick_params(axis='both', labelsize=12)
plt.show()
# fig.savefig("./Fig2/fig2_add_ACh_R2.svg", format='svg')


#%%
# ============ get spike properties ===========
props = {}
for gCa in gCa_HVAbar_nexus_list:
    props_curr = {}
    soma_v_np = dic_log[gCa]['soma_v']
    nexus_v_np = dic_log[gCa]['dend_v']

    (log,start_t,stop_t) = funs.is_calcium_spike(t_np,nexus_v_np)
    props_curr['log'] = log

    if log:
        ca_spike_dur = stop_t - start_t

        [dummy,start_ind] = funs.find_nearest(t_np,start_t)
        [dummy,stop_ind] = funs.find_nearest(t_np,stop_t)
        area = funs.calc_integral(t_np,nexus_v_np,start_ind,stop_ind,y_0)
        n_sp = funs.n_spikes(soma_v_np)

        props_curr['area'] = area
        props_curr['duration'] = ca_spike_dur
        props_curr['n_sp'] = n_sp
        props_curr['start_t'] = start_t
        props_curr['stop_t'] = stop_t

    else:
        ca_spike_dur = 0

    props[gCa] = props_curr

#%%
# ========== quantify ============
list_dur = []
for gCa in gCa_HVAbar_nexus_list:
    dur = props[gCa]['duration']
    list_dur.append(dur)

fig, ax = plt.subplots(figsize=(3.5, 1))
for i, gCa_HVA_mul in enumerate(gCa_HVA_muls):
    plt.plot(gCa_HVA_mul, list_dur[i], 'o', color=colors[i], markersize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('mS/cm^2', fontsize=16)
ax.set_ylabel('ms', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
plt.show()
# fig.savefig("./Fig2/fig2_add_ACh_R2_quantify.svg", format='svg')
