import numpy as np


def calc_conductance(type, gbar, gates):
    if type == 'HVA':
        m = gates[0]
        h = gates[1]
        g = gbar * np.power(m, 2) * h
    elif type == 'HVA_first_order':
        m = gates[0]
        h = gates[1]
        g = gbar * m * h
    elif type == 'LVA':
        m = gates[0]
        h = gates[1]
        g = gbar * np.power(m, 2) * h
    elif type == 'Ih':
        m = gates[0]
        g = gbar * m
    elif type == 'Im':
        m = gates[0]
        g = gbar * m
    elif type == 'Na':
        m = gates[0]
        h = gates[1]
        g = gbar * np.power(m, 3) * h
    elif type == 'SKv3_1':
        m = gates[0]
        g = gbar * m
    else:
        raise ValueError(f"{type} is not a valid channel name")

    return g


def calc_current(v, g, E):
    I = g * (v - E)

    return I


def calc_nernst(type, T_cel):
    R = 8.314  # J/K*mol
    T_kel = T_cel + 273.15
    if type == 'Ca':
        z = 2
        C_out = 5  # mM
        C_in = 0.0001  # mM
    if type == 'Na':
        z = 1
        C_out = 145  # mM
        C_in = 20  # mM

    if type == 'K':
        z = 1
        C_out = 5  # mM
        C_in = 140  # mM

    F = 96485  # col/mol
    E = (R * T_kel) / (z * F) * np.log(C_out / C_in) * 1000  # mV
    return E


def I_ext_constant(amp):
    return amp


def I_ext_pulse(t_curr, amp, ts=0, dur=0):
    if ts < t_curr < ts + dur:
        return amp
    else:
        return 0.0


def I_ext_syn(t_curr, amp_syn, ts=0, tau1=0.5, tau2=5):
    # synapse external current
    if t_curr < ts:
        return 0.0
    else:
        # (a)
        # I_syn = amp_syn*np.exp(-(t_curr - ts) / tau)
        # (b)
        # I_syn = amp_syn * ((t_curr - ts) / tau) *np.exp(-(t_curr - ts) / tau)
        # (c)
        I_syn = amp_syn * ((tau1*tau2)/(tau1-tau2)) *( np.exp(-(t_curr - ts) / tau1) - np.exp(-(t_curr - ts) / tau2))

        return I_syn
