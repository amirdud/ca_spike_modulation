import numpy as np

# Ca_HVA ion-channel
def Ca_HVA_m_alpha_beta(v):

    if (v == -27):
        v = v+0.0001
    alpha = (0.055 * (-27 - v)) / (np.exp((-27 - v) / 3.8) - 1)
    beta = (0.94 * np.exp((-75 - v) / 17))

    return alpha,beta

def Ca_HVA_m_inf(v):
    alpha,beta = Ca_HVA_m_alpha_beta(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf

def Ca_HVA_m_tau(v):
    alpha,beta = Ca_HVA_m_alpha_beta(v)
    m_tau = gate_tau(alpha, beta)

    return m_tau

def Ca_HVA_h_alpha_beta(v):

    if (v == -27):
        v = v + 0.0001
    alpha = (0.000457 * np.exp((-13 - v) / 50))
    beta = (0.0065 / (np.exp((-v - 15) / 28) + 1))

    return alpha, beta

def Ca_HVA_h_inf(v):
    alpha,beta = Ca_HVA_h_alpha_beta(v)
    h_inf = gate_inf(alpha, beta)

    return h_inf

def Ca_HVA_h_tau(v):
    alpha,beta = Ca_HVA_h_alpha_beta(v)
    h_tau = gate_tau(alpha, beta)

    return h_tau


def Ca_HVA_all_gates(m,h):
    # GK = (gK / Cm) * np.power(n, 4.0)

    pass



# Ca_LVA ion-channel
def Ca_LVA_m_inf(v):

    qt = np.power(2.3,((34 - 21) / 10))

    v = v + 10
    m_inf = 1.0000 / (1 + np.exp((v - -30.000) / -6))
    v = v - 10

    return m_inf

def Ca_LVA_m_tau(v):

    qt = np.power(2.3,((34 - 21) / 10))

    v = v + 10
    m_tau = (5.0000 + 20.0000 / (1 + np.exp((v - -25.000) / 5))) / qt
    v = v - 10

    return m_tau

def Ca_LVA_h_inf(v):

    qt = np.power(2.3,((34 - 21) / 10))

    v = v + 10
    h_inf = 1.0000 / (1 + np.exp((v - -80.000) / 6.4))
    v = v - 10

    return h_inf

def Ca_LVA_h_tau(v):

    qt = np.power(2.3,((34 - 21) / 10))

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

    return alpha,beta

def Ih_m_inf(v):
    alpha, beta = Ih_m_alpha_beta(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf

# Im
def Im_m_alpha_beta(v):
    alpha = 3.3e-3 * np.exp(2.5 * 0.04 * (v - -35))
    beta = 3.3e-3 * np.exp(-2.5 * 0.04 * (v - -35))
    return alpha, beta


def Im_m_alpha_beta_modified(v,shift,slope,k):
    alpha = k * np.exp(slope * 0.04 * (v - shift))
    beta = k * np.exp(-slope * 0.04 * (v - shift))
    return alpha, beta


def Im_m_inf(v):
    alpha, beta = Im_m_alpha_beta(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf

# Na
def Na_m_alpha_beta(v):
    if (v == -32):
        v = v+0.0001
    alpha = (0.182 * (v - -32)) / (1 - (np.exp(-(v - -32) / 6)))
    beta = (0.124 * (-v - 32)) / (1 - (np.exp(-(-v - 32) / 6)))

    return alpha,beta

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
    qt = np.power(2.3,((34 - 21) / 10))

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

# SK
def SK_E2_z_inf(ca):
    # if (ca < 1e-7):
    #     ca = ca + 1e-07
    # zInf = 1 / (1 + (0.00043 / ca) ^ 4.8)
    pass

# Ca synamics
def Ca_dynamics_E2(I_ca):
    gamma = 0.05 # percent of free calcium (not buffered)
    decay = 80 # (ms) rate of removal of calcium
    depth = 0.1 # (um) depth of shell
    minCai = 1e-4 # (mM)


    # cai = -(10000)*(ica*gamma/(2*FARADAY*depth)) - (cai - minCai)/decay
    pass


def calc_conductance(type,gbar,gates):

    if type == 'HVA':
        m = gates[0]
        h = gates[1]
        g = gbar * np.power(m, 2) * h
    if type == 'HVA_first_order':
        m = gates[0]
        h = gates[1]
        g = gbar * m * h
    if type == 'LVA':
        m = gates[0]
        h = gates[1]
        g = gbar * np.power(m, 2) * h
    if type == 'Ih':
        m = gates[0]
        g = gbar * m
    if type == 'Im':
        m = gates[0]
        g = gbar * m
    if type == 'Na':
        m = gates[0]
        h = gates[1]
        g = gbar * np.power(m, 3) * h
    if type == 'SKv3_1':
        m = gates[0]
        g = gbar * m

    return g


def calc_current(v, g, E):
    # g: mS/cm^2
    # v,E: mV
    # I: uA/cm^2

    I = g * (v - E)

    return I


def calc_nernst(type,T_cel):
    R = 8.314 # J/K*mol
    T_kel = T_cel + 273.15
    if type == 'Ca':
        z = 2
        C_out = 5 # mM
        C_in = 0.0001  # mM
    if type == 'Na':
        z = 1
        C_out = 145 # mM
        C_in = 20  # mM

    if type == 'K':
        z = 1
        C_out = 5 # mM
        C_in = 140  # mM

    F = 96485 # col/mol
    E = (R*T_kel)/(z*F)*np.log(C_out/C_in)*1000 # mV
    return E

def gate_inf(alpha,beta):
    gate_inf = alpha / (alpha + beta)

    return gate_inf

def gate_tau(alpha, beta):
    tau = 1 / (alpha + beta)

    return tau

def find_nearest(x,target,type='below'):
    '''find the closest value and its index above/below
        a certain number'''
    x = np.array(x)

    if type == 'above':
        inidices = np.where(x >= target)[0]
        val = np.min(x[inidices])
        idx = np.argmin(x[inidices])
    elif type == 'below':
        inidices = np.where(x <= target)[0]
        val = np.max(x[inidices])
        idx = np.argmax(x[inidices])

    return val,idx

### sympy functions
import sympy as sp
def Ca_HVA_m_inf_sympy(v):
    alpha,beta = Ca_HVA_m_alpha_beta_sympy(v)
    m_inf = gate_inf(alpha, beta)

    return m_inf

def Ca_HVA_m_alpha_beta_sympy(v):

    if (v == -27):
        v = v+0.0001
    alpha = (0.055 * (-27 - v)) / (sp.exp((-27 - v) / 3.8) - 1)
    beta = 0.94 * sp.exp((-75 - v) / 17)

    return alpha,beta
