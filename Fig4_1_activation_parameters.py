
# Channel specifications from Hay et al., 2011
# ==========================================

import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib import cm

v_rest = -75
v = np.arange(-100, 150, 1).astype(np.float32)
ind_v_rest = np.where(v == v_rest)[0][0]

# =========== Ca_HVA ===========
v[v == -27] = v[v == -27] + 0.001
mAlpha = (0.055 * (-27 - v)) / (np.exp((-27 - v) / 3.8) - 1)
mBeta = (0.94 * np.exp((-75 - v) / 17))
mInf = mAlpha / (mAlpha + mBeta)
mTau = 1 / (mAlpha + mBeta)

hAlpha = (0.000457 * np.exp((-13 - v) / 50))
hBeta = (0.0065 / (np.exp((-v - 15) / 28) + 1))
hInf = hAlpha / (hAlpha + hBeta)
hTau = 1 / (hAlpha + hBeta)

all_gates = mInf * mInf * hInf
all_gates_Ca_HVA = np.copy(all_gates)

mInf_Ca_HVA = np.copy(mInf)
mTau_Ca_HVA = np.copy(mTau)
hInf_Ca_HVA = np.copy(hInf)
hTau_Ca_HVA = np.copy(hTau)

# =========== Im ===========
shift = -39
slope = 2.3
k =2.0e-3

qt = math.pow(2.3, ((34 - 21) / 10))
v = np.arange(-100, 150, 1).astype(np.float32)
mAlpha = k * np.exp(slope * 0.04 * (v - shift))
mBeta = k * np.exp(-slope * 0.04 * (v - shift))

mInf = mAlpha / (mAlpha + mBeta)
mTau = (1 / (mAlpha + mBeta)) / qt

all_gates = mInf
all_gates_Im = np.copy(all_gates)
mInf_Im = np.copy(mInf)
mTau_Im = np.copy(mTau)

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
# fig.savefig("./Fig4/fig4_hTau_Ca_HVA.svg", format='svg')

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
# fig.savefig("./Fig4/fig4_mTau_Ca_HVA_Im.svg", format='svg')


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
# fig.savefig("./Fig4/fig4_m_Ca_HVA_Im.svg", format='svg')