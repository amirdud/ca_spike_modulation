import matplotlib.pyplot as plt
import channel_funs as cf
from matplotlib import cm
from cfg import *

general_params = cfg_fig_4_1["h"]
v = general_params["v"]

Im_params = cfg_fig_4_1["Im"]
shift = Im_params["shift"]
slope = Im_params["slope"]
k = Im_params["k"]

# =========== Ca_HVA ===========
mAlpha_CaHVA = []
mBeta_CaHVA = []
hAlpha_CaHVA = []
hBeta_CaHVA = []
mAlpha_Im = []
mBeta_Im = []

for vi in v:
    mAlpha_CaHVA_i, mBeta_CaHVA_i = cf.Ca_HVA_m_alpha_beta(vi)
    hAlpha_CaHVA_i, hBeta_CaHVA_i = cf.Ca_HVA_h_alpha_beta(vi)
    mAlpha_Im_i, mBeta_Im_i = cf.Im_m_alpha_beta_with_params(vi, shift, slope, k)

    mAlpha_CaHVA.append(mAlpha_CaHVA_i)
    mBeta_CaHVA.append(mBeta_CaHVA_i)
    hAlpha_CaHVA.append(hAlpha_CaHVA_i)
    hBeta_CaHVA.append(hBeta_CaHVA_i)
    mAlpha_Im.append(mAlpha_Im_i)
    mBeta_Im.append(mBeta_Im_i)

mAlpha_CaHVA = np.array(mAlpha_CaHVA)
mBeta_CaHVA = np.array(mBeta_CaHVA)
hAlpha_CaHVA = np.array(hAlpha_CaHVA)
hBeta_CaHVA = np.array(hBeta_CaHVA)
mAlpha_Im = np.array(mAlpha_Im)
mBeta_Im = np.array(mBeta_Im)

mInf_Ca_HVA = cf.gate_inf(mAlpha_CaHVA, mBeta_CaHVA)
hInf_Ca_HVA = cf.gate_inf(hAlpha_CaHVA, hBeta_CaHVA)
mInf_Im = cf.gate_inf(mAlpha_Im, mBeta_Im)

mTau_Ca_HVA = cf.gate_tau(mAlpha_CaHVA, mBeta_CaHVA)
hTau_Ca_HVA = cf.gate_tau(hAlpha_CaHVA, hBeta_CaHVA)
mTau_Im = cf.Im_m_tau(mAlpha_Im, mBeta_Im)

# ========= time constants ==========
evenly_spaced_interval = np.linspace(0, 1, 7)
colors = [cm.coolwarm(k) for k in evenly_spaced_interval]

fig, ax = plt.subplots(figsize=(1.5, 2))
ax.plot(v, hTau_Ca_HVA, '--', color=colors[6])
ax.set_xlabel('mV', fontsize=16)
ax.set_ylabel('tau (ms)', fontsize=16)
ax.set_ylim([0, 500])
ax.set_xlim([-100, 50])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig4/fig4_hTau_Ca_HVA.svg", format='svg')

fig, ax = plt.subplots(figsize=(1.5, 2))
ax.plot(v, mTau_Ca_HVA, color=colors[6])
ax.plot(v, mTau_Im, color=colors[0])
ax.set_xlabel('mV', fontsize=16)
ax.set_ylabel('tau (ms)', fontsize=16)
ax.set_xlim([-100, 50])
ax.set_ylim([0, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig4/fig4_mTau_Ca_HVA_Im.svg", format='svg')


# ========= activation curves ==========
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(v, mInf_Im, linewidth=4, color=colors[0])
ax.plot(v, mInf_Ca_HVA, linewidth=4, color=colors[6])
ax.set_xlabel('mV', fontsize=16)
ax.set_ylabel('m_inf', fontsize=16)
ax.set_ylim([0, 1])
ax.set_xlim([-60, 0])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
if general_params["save_figs"]:
    fig.savefig("./Fig4/fig4_m_Ca_HVA_Im.svg", format='svg')
