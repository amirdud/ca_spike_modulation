#%%

import numpy as np
from neuron import h,gui
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

##=================== creating cell object ===========================

h.load_file("import3d.hoc")
morphology_file = "morphologies/cell1.asc"
h.load_file("models/L5PCbiophys5b.hoc")
h.load_file("models/L5PCtemplate_2.hoc")

h.define_shape()
L5PC = h.L5PCtemplate(morphology_file)

## ============== parameters =================

syn_1 = h.epsp(L5PC.apic[36](0.9))
syn_1.tau0 = 0.5
syn_1.tau1 = 5
syn_1.onset = 155
syn_1.imax = 1.6

h.tstop = 300
h.v_init = -80
h.celsius = 37

## ============== variables =================
t = h.Vector()
soma_v = h.Vector()
dend_v = h.Vector()
syn_1_i = h.Vector()
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
syn_1_i.record(syn_1._ref_i)
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

## ============== simulation =================
h.run()
soma_v_np = np.array(soma_v)
dend_v_np = np.array(dend_v)
syn_1_i_np = np.array(syn_1_i)*(-1)
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
t_np = np.array(t)
t_np_shifted = t_np - syn_1.onset


## =========== plot =========

f,ax1 = plt.subplots(figsize=(4,2))
ax1.plot(t_np_shifted,soma_v_np,'k',label='soma_v')
ax1.plot(t_np_shifted,dend_v_np,'r',label='dend_v')
ax1.set_ylabel('mV',fontsize = 16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_ylim([-90,50])
ax1.set_xlim([-10,70])

plt.show()
# f.savefig("Fig1/fig1_EPSP_Ca_Spike.svg", format='svg')

f,ax2 = plt.subplots(figsize=(4,0.5))
ax2.plot(t_np_shifted,syn_1_i_np,'r',label='I')
ax2.set_ylabel('nA',fontsize = 16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim([0,2.5])
ax2.set_xlim([-10,70])
ax2.set_xlabel('ms',fontsize = 16)

plt.show()
# f.savefig("Fig1/fig1_EPSP_current.svg", format='svg')
# f.savefig("Fig1/fig1_BAC_Ca_Spike.svg", format='svg')
