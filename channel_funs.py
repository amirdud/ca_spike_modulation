import math
import numpy as np
import sympy as sp

# Channel specifications from Hay et al., 2011
# ==========================================

# Ca_HVA ion-channel
def Ca_HVA_m_alpha_beta(v):
    if (v == -27):
        v = v + 0.0001
    alpha = (0.055 * (-27 - v)) / (np.exp((-27 - v) / 3.8) - 1)
    beta = (0.94 * np.exp((-75 - v) / 17))

    return alpha, beta


def Ca_HVA_m_inf(v):
    alpha, beta = Ca_HVA_m_alpha_beta(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf


def Ca_HVA_m_tau(v):
    alpha, beta = Ca_HVA_m_alpha_beta(v)
    m_tau = gate_tau(alpha, beta)

    return m_tau


def Ca_HVA_h_alpha_beta(v):
    if (v == -27):
        v = v + 0.0001
    alpha = (0.000457 * np.exp((-13 - v) / 50))
    beta = (0.0065 / (np.exp((-v - 15) / 28) + 1))

    return alpha, beta


def Ca_HVA_h_inf(v):
    alpha, beta = Ca_HVA_h_alpha_beta(v)
    h_inf = gate_inf(alpha, beta)

    return h_inf


def Ca_HVA_h_tau(v):
    alpha, beta = Ca_HVA_h_alpha_beta(v)
    h_tau = gate_tau(alpha, beta)

    return h_tau


# Ca_LVA ion-channel
def Ca_LVA_m_inf(v):

    v = v + 10
    m_inf = 1.0000 / (1 + np.exp((v - -30.000) / -6))
    v = v - 10

    return m_inf


def Ca_LVA_m_tau(v):
    qt = np.power(2.3, ((34 - 21) / 10))

    v = v + 10
    m_tau = (5.0000 + 20.0000 / (1 + np.exp((v - -25.000) / 5))) / qt
    v = v - 10

    return m_tau


def Ca_LVA_h_inf(v):

    v = v + 10
    h_inf = 1.0000 / (1 + np.exp((v - -80.000) / 6.4))
    v = v - 10

    return h_inf


def Ca_LVA_h_tau(v):
    qt = np.power(2.3, ((34 - 21) / 10))

    v = v + 10
    h_tau = (20.0000 + 50.0000 / (1 + np.exp((v - -40.000) / 7))) / qt
    v = v - 10

    return h_tau


# Ih
def Ih_m_alpha_beta(v):
    vhalf = -154.9
    vshift = 0

    if v == vhalf:
        v = v + 0.0001

    alpha = 0.001 * 6.43 * (v - (vhalf + vshift)) / (np.exp((v - (vhalf + vshift)) / 11.9) - 1)
    beta = 0.001 * 193 * np.exp((v - vshift) / 33.1)

    return alpha, beta


def Ih_m_inf(v):
    alpha, beta = Ih_m_alpha_beta(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf


# Im
def Im_m_alpha_beta(v):
    alpha = 3.3e-3 * np.exp(2.5 * 0.04 * (v - -35))
    beta = 3.3e-3 * np.exp(-2.5 * 0.04 * (v - -35))
    return alpha, beta


def Im_m_alpha_beta_with_params(v, shift, slope, k):
    alpha = k * np.exp(slope * 0.04 * (v - shift))
    beta = k * np.exp(-slope * 0.04 * (v - shift))
    return alpha, beta


def Im_m_inf(v):
    alpha, beta = Im_m_alpha_beta(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf


def Im_m_tau(alpha, beta):
    qt = math.pow(2.3, ((34 - 21) / 10))
    tau = (1 / (alpha + beta)) / qt

    return tau


# Na
def Na_m_alpha_beta(v):
    if (v == -32):
        v = v + 0.0001
    alpha = (0.182 * (v - -32)) / (1 - (np.exp(-(v - -32) / 6)))
    beta = (0.124 * (-v - 32)) / (1 - (np.exp(-(-v - 32) / 6)))

    return alpha, beta


def Na_m_inf(v):
    alpha, beta = Na_m_alpha_beta(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf


def Na_h_alpha_beta(v):
    if (v == -60):
        v = v + 0.0001
    alpha = (-0.015 * (v - -60)) / (1 - (np.exp((v - -60) / 6)))
    beta = (-0.015 * (-v - 60)) / (1 - (np.exp((-v - 60) / 6)))

    return alpha, beta


def Na_h_inf(v):
    alpha, beta = Na_h_alpha_beta(v)
    h_inf = gate_inf(alpha, beta)

    return h_inf


def Na_h_tau(v):
    qt = np.power(2.3, ((34 - 21) / 10))

    alpha, beta = Na_h_alpha_beta(v)
    h_tau = (1 / (alpha + beta)) / qt

    return h_tau


# SKv3_1
def SKv3_1_m_inf(v):
    m_inf = 1 / (1 + np.exp(((v - (18.700)) / (-9.700))))

    return m_inf


def SKv3_1_m_tau(v):
    m_tau = 0.2 * 20.000 / (1 + np.exp(((v - (-46.560)) / (-44.140))))

    return m_tau


def Ca_HVA_m_inf_sympy(v):
    alpha, beta = Ca_HVA_m_alpha_beta_sympy(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf


def Ca_HVA_m_alpha_beta_sympy(v):
    if (v == -27):
        v = v + 0.0001
    alpha = (0.055 * (-27 - v)) / (sp.exp((-27 - v) / 3.8) - 1)
    beta = 0.94 * sp.exp((-75 - v) / 17)

    return alpha, beta


def gate_inf(alpha, beta):
    gate_inf = alpha / (alpha + beta)

    return gate_inf


def gate_tau(alpha, beta):
    tau = 1 / (alpha + beta)

    return tau

# HH K
def HH_K_alpha_n(Vm):
    return (0.01 * (Vm + 55.0)) / (1 - np.exp(-(Vm + 55.0)/10.0))


def HH_K_beta_n(Vm):
    return (0.125 * np.exp(-(Vm + 65.0)/80.0))


# HH Na
def HH_Na_alpha_m(Vm):
    return (0.1 * (Vm + 40.0)) / (1 - np.exp(-(Vm + 40.0)/10.0))


def HH_Na_beta_m(Vm):
    return (4 * np.exp(-(Vm + 65.0)/18.0))


def HH_Na_alpha_h(Vm):
    return 0.07 * np.exp(-(Vm + 65.0) / 20.0)


def HH_Na_beta_h(Vm):
    return (1 / (1 + np.exp(-(Vm + 35.0) / 10.0)) )


def HH_K_n_inf(Vm=-65.0):
    return HH_K_alpha_n(Vm) / (HH_K_alpha_n(Vm) + HH_K_beta_n(Vm))


def HH_Na_m_inf(Vm=-65.0):
    return HH_Na_alpha_m(Vm) / (HH_Na_alpha_m(Vm) + HH_Na_beta_m(Vm))


def HH_Na_h_inf(Vm=-65.0):
    return HH_Na_alpha_h(Vm) / (HH_Na_alpha_h(Vm) + HH_Na_beta_h(Vm))