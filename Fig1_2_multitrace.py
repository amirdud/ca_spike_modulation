
import time
import numpy as np
from neuron import h,gui
import matplotlib.pyplot as plt
import matplotlib
import funs
import pickle
import seaborn as sns

matplotlib.use('Qt5Agg')

##=================== creating cell object ===========================

h.load_file("import3d.hoc")
morphology_file = "morphologies/cell1.asc"
h.load_file("models/L5PCbiophys5b.hoc")
h.load_file("models/L5PCtemplate_2.hoc")

h.define_shape()
L5PC = h.L5PCtemplate(morphology_file)

## ============== parameters =================
stim = h.IClamp(L5PC.soma[0](0.5))
stim.delay = 150
amps = np.round(np.arange(0,5.1,2),1)
durs = np.round(np.arange(4,6,1),1)

syn_1 = h.epsp(L5PC.apic[36](0.9))
tau0s = np.round(np.arange(0.5,0.6,0.1),1)
tau1s = np.round(np.arange(5,6,3),1)
BACdts = np.round(np.arange(0,5+2.5,2.5),1)
imaxs = np.round(np.arange(0,3.1,1),1)

h.tstop = 250
h.v_init = -80
h.celsius = 37
t_np = np.arange(0,h.tstop+0.025,0.025)

run_sim = False

## ============== variables =================
t = h.Vector()
soma_v = h.Vector()
dend_v = h.Vector()
stim_i = h.Vector()
syn_1_i = h.Vector()
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
gHVA_ca = h.Vector()
gLVA_ca = h.Vector()
gSK_k = h.Vector()
gSKv3_k = h.Vector()
gIm_k = h.Vector()
gIh_cn = h.Vector()
gNaTs2_t_na = h.Vector()
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
soma_v.record(L5PC.soma[0](0.5)._ref_v)
dend_v.record(L5PC.apic[36](0.9)._ref_v)
stim_i.record(stim._ref_i)
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
gHVA_ca.record(L5PC.apic[36](0.9).Ca_HVA._ref_gCa_HVA)
gLVA_ca.record(L5PC.apic[36](0.9).Ca_LVAst._ref_gCa_LVAst)
gSK_k.record(L5PC.apic[36](0.9).SK_E2._ref_gSK_E2)
gSKv3_k.record(L5PC.apic[36](0.9).SKv3_1._ref_gSKv3_1)
gIm_k.record(L5PC.apic[36](0.9).Im._ref_gIm)
gIh_cn.record(L5PC.apic[36](0.9).Ih._ref_gIh)
gNaTs2_t_na.record(L5PC.apic[36](0.9).NaTs2_t._ref_gNaTs2_t)
mHVA_ca.record(L5PC.apic[36](0.9).Ca_HVA._ref_m)
hHVA_ca.record(L5PC.apic[36](0.9).Ca_HVA._ref_h)
mLVA_ca.record(L5PC.apic[36](0.9).Ca_LVAst._ref_m)
hLVA_ca.record(L5PC.apic[36](0.9).Ca_LVAst._ref_h)
zSK_k.record(L5PC.apic[36](0.9).SK_E2._ref_z)
mSKv3_k.record(L5PC.apic[36](0.9).SKv3_1._ref_m)
mIm_k.record(L5PC.apic[36](0.9).Im._ref_m)
mIh_cn.record(L5PC.apic[36](0.9).Ih._ref_m)
mNaTs2_t_na.record(L5PC.apic[36](0.9).NaTs2_t._ref_m)
hNaTs2_t_na.record(L5PC.apic[36](0.9).NaTs2_t._ref_h)


## ============== simulation =================

if run_sim:
    print('Running Simulation...')
    dic_log = {}
    for amp in amps:
        for dur in durs:
            for tau0 in tau0s:
                for tau1 in tau1s:
                    for BACdt in BACdts:
                        for imax in imaxs:
                            tic = time.time()
                            data = {}

                            # stim params
                            stim.amp = amp
                            stim.dur = dur

                            # syn_1 params
                            syn_1.tau0 = tau0
                            syn_1.tau1 = tau1
                            syn_1.onset = stim.delay + dur + BACdt
                            syn_1.imax = imax


                            h.run()

                            soma_v_np = np.array(soma_v)
                            dend_v_np = np.array(dend_v)
                            stim_i_np = np.array(stim_i)
                            syn_1_i_np = np.array(syn_1_i)

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
                            gHVA_ca_np = np.array(gHVA_ca)
                            gLVA_ca_np = np.array(gLVA_ca)
                            gSK_k_np = np.array(gSK_k)
                            gSKv3_k_np = np.array(gSKv3_k)
                            gIm_k_np = np.array(gIm_k)
                            gIh_cn_np = np.array(gIh_cn)
                            gNaTs2_t_na_np = np.array(gNaTs2_t_na)
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

                            data['soma_v'] = soma_v_np
                            data['dend_v'] = dend_v_np
                            data['stim_i'] = stim_i_np
                            data['syn_1_i'] = syn_1_i_np
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
                            data['gHVA_ca'] = gHVA_ca_np
                            data['gLVA_ca'] = gLVA_ca_np
                            data['gSK_k'] = gSK_k_np
                            data['gSKv3_k'] = gSKv3_k_np
                            data['gIm_k'] = gIm_k_np
                            data['gIh_cn'] = gIh_cn_np
                            data['gNaTs2_t_na'] = gNaTs2_t_na_np
                            data['mHVA_ca'] = mHVA_ca_np
                            data['hHVA_ca'] = hHVA_ca_np
                            data['mLVA_ca'] = mLVA_ca_np
                            data['hLVA_ca'] = hLVA_ca_np
                            data['zSK_k'] = zSK_k_np
                            data['mSKv3_k'] = mSKv3_k_np
                            data['mIm_k'] = mIm_k_np
                            data['mIh_cn'] = mIh_cn_np
                            data['mNaTs2_t_na'] = mNaTs2_t_na_np
                            data['hNaTs2_t_na'] = hNaTs2_t_na_np

                            dic_log[(amp,dur,tau0,tau1,BACdt,imax)] = data
                            toc = time.time()

                            print(f'finished: amp = {amp:.2f}, dur = {dur:.2f}, '
                                  f'tau0 = {tau0:.2f}, tau1 = {tau1:.2f}, '
                                  f'BACdt = {BACdt:.2f}, imax = {imax:.2f} '
                                  f'in {toc-tic:.2f} sec')

    # save
    filename = 'dic_log_L5PC_traces_fig1.pickle'
    pickle.dump(dic_log, open(filename, 'wb'), protocol=2)

else:
    # load
    print('Loading...')
    # filename = 'dic_log_L5PC_traces_fig1.pickle'
    filename = 'dic_log_L5PC_exploration_BAC_fig1_many_traces.pickle'
    file = open(filename,'rb')
    dic_log = pickle.load(file)
    file.close()

#%%
## ========= get spike paramters ========

props = {}
for amp in amps:
    for dur in durs:
        for tau0 in tau0s:
            for tau1 in tau1s:
                for BACdt in BACdts:
                    for imax in imaxs:
                        props_curr = {}
                        soma_v_np = dic_log[(amp,dur,tau0,tau1,BACdt,imax)]['soma_v']
                        nexus_v_np = dic_log[(amp,dur,tau0,tau1,BACdt,imax)]['dend_v']

                        (log,start_t,stop_t) = funs.is_calcium_spike(t_np,nexus_v_np)
                        props_curr['log'] = log

                        if log:
                            ca_spike_dur = stop_t - start_t

                            [dummy,start_ind] = funs.find_nearest(t_np,start_t)
                            [dummy,stop_ind] = funs.find_nearest(t_np,stop_t)
                            n_sp = funs.n_spikes(soma_v_np)

                            ind_peak = np.argmax(nexus_v_np)

                            props_curr['duration'] = ca_spike_dur
                            props_curr['n_sp'] = n_sp
                            props_curr['start_t'] = start_t
                            props_curr['stop_t'] = stop_t
                            props_curr['ind_peak'] = ind_peak

                        else:
                            ca_spike_dur = 0

                        props[(amp,dur,tau0,tau1,BACdt,imax)] = props_curr

#%%
## =========== plot all =============

fig, ax = plt.subplots()
for amp in amps:
    for dur in durs:
        for tau0 in tau0s:
            for tau1 in tau1s:
                for BACdt in BACdts:
                    for imax in imaxs:


                        ax.plot(t_np, dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['dend_v'], 'k',label='dend_v')
                        ax.set_ylabel('mV')
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

plt.show()

#%%
## =========== barplot =============
bef_peak_t = 20 # msec
aft_peak_t = 60 # msec
bef_peak_ind = int(bef_peak_t/h.dt)
aft_peak_ind = int(aft_peak_t/h.dt)

t4 = 31
ind4 = int(t4/h.dt)

t3 = 21
ind3 = int(t3/h.dt)

t2 = 11
ind2 = int(t2/h.dt)

t1 = 1
ind1 = int(t1/h.dt)


dend_v_shifted_all = []
m_dvdt_dend_v_P1 = []
m_dvdt_dend_v_P2 = []
m_dvdt_dend_v_P3 = []

dvdt_dend_v_P1 = []
dvdt_dend_v_P2 = []
dvdt_dend_v_P3 = []

for amp in amps:
    for dur in durs:
        for tau0 in tau0s:
            for tau1 in tau1s:
                for BACdt in BACdts:
                    for imax in imaxs:
                        try:
                            ind_peak = props[(amp, dur, tau0, tau1, BACdt, imax)]['ind_peak']
                        except:
                            continue
                        t_shifted = t_np[ind_peak - bef_peak_ind:ind_peak + aft_peak_ind] - t_np[ind_peak:][0]
                        dend_v_shifted = dic_log[(amp,dur,tau0,tau1,BACdt,imax)]['dend_v'][ind_peak-bef_peak_ind:ind_peak+aft_peak_ind]
                        soma_v_shifted = dic_log[(amp,dur,tau0,tau1,BACdt,imax)]['soma_v'][ind_peak-bef_peak_ind:ind_peak+aft_peak_ind]

                        dvdt_dend_v_shifted = np.diff(dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['dend_v'][ind_peak - bef_peak_ind:ind_peak + aft_peak_ind])
                        dvdt_dend_v_P1_i = np.diff(dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['dend_v'][ind_peak - ind2:ind_peak - ind1]).reshape((1,-1))
                        dvdt_dend_v_P2_i = np.diff(dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['dend_v'][ind_peak + ind1:ind_peak + ind2]).reshape((1,-1))
                        dvdt_dend_v_P3_i = np.diff(dic_log[(amp, dur, tau0, tau1, BACdt, imax)]['dend_v'][ind_peak + ind3:ind_peak + ind4]).reshape((1,-1))

                        dvdt_dend_v_P1.append(dvdt_dend_v_P1_i)
                        dvdt_dend_v_P2.append(dvdt_dend_v_P2_i)
                        dvdt_dend_v_P3.append(dvdt_dend_v_P3_i)

dvdt_dend_v_P1_np = np.concatenate(dvdt_dend_v_P1, axis=0)
dvdt_dend_v_P2_np = np.concatenate(dvdt_dend_v_P2, axis=0)
dvdt_dend_v_P3_np = np.concatenate(dvdt_dend_v_P3, axis=0)

P1_std = np.std(dvdt_dend_v_P1_np,axis=0)
P2_std = np.std(dvdt_dend_v_P2_np,axis=0)
P3_std = np.std(dvdt_dend_v_P3_np,axis=0)

fig,(ax0,ax1,ax2)=plt.subplots(1,3)
ax0.plot(P1_std)
ax1.plot(P2_std)
ax2.plot(P3_std)
plt.show()


#%% barplot
m_dvdt_dend_v = [P1_std,P2_std,P3_std]
xs = [np.ones(P1_std.shape),2*np.ones(P1_std.shape),3*np.ones(P3_std.shape)]
ave_xs = [1,2,3]
ave_m_dvdt_dend_v = [P1_std.mean(),P2_std.mean(),P3_std.mean()]
std_m_dvdt_dend_v = [P1_std.std(),P2_std.std(),P3_std.std()]

fig,ax = plt.subplots(figsize=(1.2,1.5))
medianprops = {'color':'w'}
colors = ['k','k','k']
barp = ax.bar(ave_xs,ave_m_dvdt_dend_v,yerr=std_m_dvdt_dend_v,color='k')

plt.show()
# fig.savefig("Fig1/fig1_boxplot.svg", format='svg')
