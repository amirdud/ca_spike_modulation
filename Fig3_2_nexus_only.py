from neuron import h,gui
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np

matplotlib.use('Qt5Agg')


##=================== creating cell object ===========================
# read nexus parameters
file = open("nexus_point_neuron_parameters.pkl",'rb')
nexus_parameters = pickle.load(file)
file.close()

soma = h.Section(name='soma')

soma.L = 10
soma.diam = 10
soma.insert('pas')
soma(0.5).g_pas = nexus_parameters['g_pas']
soma(0.5).e_pas = nexus_parameters['e_pas']

soma.insert('CaDynamics_E2')
soma(0.5).gamma_CaDynamics_E2 = nexus_parameters['gamma_CaDynamics_E2']
soma(0.5).decay_CaDynamics_E2 = nexus_parameters['decay_CaDynamics_E2']

soma.insert('Ca_HVA')
soma(0.5).gCa_HVAbar_Ca_HVA = nexus_parameters['gCa_HVAbar_Ca_HVA']

soma.insert('Ca_LVAst')
soma(0.5).gCa_LVAstbar_Ca_LVAst = nexus_parameters['gCa_LVAstbar_Ca_LVAst']

soma.insert('Ih')
soma(0.5).gIhbar_Ih = nexus_parameters['gIhbar_Ih']

soma.insert('Im')
soma(0.5).gImbar_Im = nexus_parameters['gImbar_Im']

soma.insert('NaTs2_t')
soma(0.5).gNaTs2_tbar_NaTs2_t = nexus_parameters['gNaTs2_tbar_NaTs2_t']

soma.insert('SK_E2')
soma(0.5).gSK_E2bar_SK_E2 = nexus_parameters['gSK_E2bar_SK_E2']

soma.insert('SKv3_1')
soma(0.5).gSKv3_1bar_SKv3_1 = nexus_parameters['gSKv3_1bar_SKv3_1']


# =========== variables =============
t = h.Vector()
soma_v = h.Vector()
stim_i = h.Vector()
HVA_ca = h.Vector()
LVA_ca = h.Vector()
SK_k = h.Vector()
SKv3_k = h.Vector()
Im_k = h.Vector()
Ih_cn = h.Vector()
NaTs2_t_na = h.Vector()
ik = h.Vector()
ina = h.Vector()
ica = h.Vector()

gHVA = h.Vector()
gLVA = h.Vector()
gSK = h.Vector()
gSKv3 = h.Vector()
gIm = h.Vector()
gIh = h.Vector()
gNaTs2_t = h.Vector()

mHVA_ca = h.Vector()
hHVA_ca = h.Vector()
mLVA_ca = h.Vector()
hLVA_ca = h.Vector()
zSK_k = h.Vector()
mSKv3_k = h.Vector()
mIm_k = h.Vector()
mIh_cn = h.Vector()
mNaTs2_t_na = h.Vector()
hNaTs2_t_na = h.Vector()

t.record(h._ref_t)
soma_v.record(soma(0.5)._ref_v)
HVA_ca.record(soma(0.5).Ca_HVA._ref_ica)
LVA_ca.record(soma(0.5).Ca_LVAst._ref_ica)
SK_k.record(soma(0.5).SK_E2._ref_ik)
SKv3_k.record(soma(0.5).SKv3_1._ref_ik)
Im_k.record(soma(0.5).Im._ref_ik)
Ih_cn.record(soma(0.5).Ih._ref_ihcn)
NaTs2_t_na.record(soma(0.5).NaTs2_t._ref_ina)
ik.record(soma(0.5).k_ion._ref_ik)
ica.record(soma(0.5).ca_ion._ref_ica)
ina.record(soma(0.5).na_ion._ref_ina)
gHVA.record(soma(0.5).Ca_HVA._ref_gCa_HVA)
gLVA.record(soma(0.5).Ca_LVAst._ref_gCa_LVAst)
gSK.record(soma(0.5).SK_E2._ref_gSK_E2)
gSKv3.record(soma(0.5).SKv3_1._ref_gSKv3_1)
gIm.record(soma(0.5).Im._ref_gIm)
gIh.record(soma(0.5).Ih._ref_gIh)
gNaTs2_t.record(soma(0.5).NaTs2_t._ref_gNaTs2_t)
mHVA_ca.record(soma(0.5).Ca_HVA._ref_m)
hHVA_ca.record(soma(0.5).Ca_HVA._ref_h)
mLVA_ca.record(soma(0.5).Ca_LVAst._ref_m)
hLVA_ca.record(soma(0.5).Ca_LVAst._ref_h)
zSK_k.record(soma(0.5).SK_E2._ref_z)
mSKv3_k.record(soma(0.5).SKv3_1._ref_m)
mIm_k.record(soma(0.5).Im._ref_m)
mIh_cn.record(soma(0.5).Ih._ref_m)
mNaTs2_t_na.record(soma(0.5).NaTs2_t._ref_m)
hNaTs2_t_na.record(soma(0.5).NaTs2_t._ref_h)

# ============== parameters =================
h.tstop = 250
h.v_init = -80
h.celsius = 37

syn_1 = h.epsp(soma(0.5))
syn_1.tau0 = 0.5
syn_1.tau1 = 5
syn_1.onset = 150
syn_1.imax = 0.03
stim_i.record(syn_1._ref_i)

# ============== simulation =================
h.run()
soma_v_np = np.array(soma_v)
stim_i_np = np.array(stim_i)
HVA_ca_np = np.array(HVA_ca)
LVA_ca_np = np.array(LVA_ca)
SK_k_np = np.array(SK_k)
SKv3_k_np = np.array(SKv3_k)
Im_k_np = np.array(Im_k)
Ih_cn_np = np.array(Ih_cn)
NaTs2_t_na_np = np.array(NaTs2_t_na)
ik_np = np.array(ik)
ica_np = np.array(ica)
ina_np = np.array(ina)
gHVA_np = np.array(gHVA)
gLVA_np = np.array(gLVA)
gSK_np = np.array(gSK)
gSKv3_np = np.array(gSKv3)
gIm_np = np.array(gIm)
gIh_np = np.array(gIh)
gNaTs2_t_np = np.array(gNaTs2_t)
mHVA_ca_np = np.array(mHVA_ca)
hHVA_ca_np = np.array(hHVA_ca)
mLVA_ca_np = np.array(mLVA_ca)
hLVA_ca_np = np.array(hLVA_ca)
zSK_k_np = np.array(zSK_k)
mSKv3_k_np = np.array(mSKv3_k)
mIm_k_np = np.array(mIm_k)
mIh_cn_np = np.array(mIh_cn)
mNaTs2_t_na_np = np.array(mNaTs2_t_na)
hNaTs2_t_na_np = np.array(hNaTs2_t_na)

t_np = np.array(t)
t_np_shifted = t_np - syn_1.onset

# ========= order currents ==========
HVA_ca_pre_max = max(abs(HVA_ca_np))
LVA_ca_pre_max = max(abs(LVA_ca_np))
SK_k_max = max(abs(SK_k_np))
SKv3_k_max = max(abs(SKv3_k_np))
Im_k_max = max(abs(Im_k_np))
Ih_cn_max = max(abs(Ih_cn_np))
NaTs2_t_max = max(abs(NaTs2_t_na_np))

# max values (+ sign)
max_vals = [HVA_ca_pre_max,
            LVA_ca_pre_max,
            SK_k_max * (-1),
            SKv3_k_max * (-1),
            Im_k_max * (-1),
            Ih_cn_max,
            NaTs2_t_max]

max_vals_sorted_inds = np.argsort(max_vals)

# =========== plot ==========
from matplotlib import cm

evenly_spaced_interval = np.linspace(0, 1, 7)
colors = [cm.coolwarm(k) for k in evenly_spaced_interval]

fig, ax0 = plt.subplots(figsize=(4, 4))
line_Im, = ax0.plot(t_np_shifted, Im_k_np, label='Im', color=colors[0])
ax0.plot(t_np_shifted, SKv3_k_np, label='SKv3', color=colors[1])
ax0.plot(t_np_shifted, SK_k_np, label='SK', color=colors[2])
ax0.plot(t_np_shifted, Ih_cn_np, label='Ih', color=colors[3])
ax0.plot(t_np_shifted, LVA_ca_np, label='iLVA_ca', color=colors[4])
ax0.plot(t_np_shifted, NaTs2_t_na_np, label='Na', color=colors[5])
ax0.plot(t_np_shifted, HVA_ca_np, label='iHVA_ca', color=colors[6])
ax0.set_ylabel('mA/cm2', fontsize=16)
ax0.set_xlabel('ms', fontsize=16)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.set_ylim([-0.1, 0.06])
ax0.set_xlim([-10, 70])
ax0.set_yticks(np.arange(-0.1, 0.06 + 0.04, 0.04))
plt.show()
# fig.savefig("./Fig3/fig3_nexusoma_currents.svg", format='svg')

fig, ax1 = plt.subplots(figsize=(4, 2))
ax1.plot(t_np_shifted, soma_v, 'r', label='nexusoma_v')
ax1.set_ylabel('mV', fontsize=16)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_ylim([-90, 50])
ax1.set_xlim([-10, 70])
plt.show()
# fig.savefig("./Fig3/fig3_nexusoma_spike.svg", format='svg')

fig, ax2 = plt.subplots(figsize=(4, 0.5))
ax2.plot(t_np_shifted, stim_i_np * (-1), 'r', label='I')
ax2.set_ylabel('nA', fontsize=16)
ax2.set_xlabel('ms', fontsize=16)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim([0, 0.05])
ax2.set_xlim([-10, 70])
plt.show()
# fig.savefig("./Fig3/fig3_nexusoma_epsp.svg", format='svg')