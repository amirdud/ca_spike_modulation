import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import random
import scipy.signal
import mod_dynamics_funs as mdf

def get_random_segments(sections,n_segs,seed=False):
    '''
    get a list of random segments from a list of sections (from a neuron model)

    sections: list of sections from a neuron model
    n_segs: number of segments to return from all these sections

    return: segments_subsample: a list of segments from all spiecified sections
    '''
    segments=[]

    for sec in sections:
        for seg in sec:
            segments.append(seg)

    if isinstance(seed,int):
        random.seed(seed)

    segments_subsample = random.sample(segments,n_segs)

    return segments_subsample


def find_nearest(x,target,type='below'):
    '''
    find the closest value and its index above/below
        a certain number'

    x: np.array of values
    target: target value to find the closest value to it
    type: find the closest below/above the target value

    val: the closest value
    idx: the index of the closest value

    '''

    if type == 'above':
        inidices = np.where(x >= target)[0]
        val = np.min(x[inidices])
        idx = inidices[0]

    elif type == 'below':
        inidices = np.where(x <= target)[0]
        val = np.max(x[inidices])
        idx = np.argmax(x[inidices])

    return val,idx


def calc_integral(x, y, start_ind=0, stop_ind=-1, y_0=0, show=False):
    y_use = y[start_ind:stop_ind] - y_0
    x_use = x[start_ind:stop_ind]

    area = np.trapz(y_use,x_use)

    if show:
        f,ax = plt.subplots()
        ax.plot(x_use,y_use)
        ax.set_title(area)

    return area


def n_spikes(soma_v_np):
    th_sp = 0.15 # mV/ms
    diff_soma_v = np.diff(soma_v_np)
    smooth_diff_soma_v = smooth(diff_soma_v, 35)
    loc_peaks = scipy.signal.find_peaks(smooth_diff_soma_v,height=th_sp)[0]
    n_sp = loc_peaks.size
    return n_sp


def time_first_spike(soma_v_np,Fs):
    th_sp = 0.15 # mV/ms
    diff_soma_v = np.diff(soma_v_np)
    smooth_diff_soma_v = smooth(diff_soma_v, 35)
    loc_peaks = scipy.signal.find_peaks(smooth_diff_soma_v,height=th_sp)[0]
    timing_1st = loc_peaks[0]/Fs
    return timing_1st


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def find_graph_intersections(f,g):
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()
    return indices


def calc_spike_duration(t,v):
    mV_th = -70
    bool_v = v > mV_th
    diff_bool_v = np.diff(bool_v)
    sp_dur = t[:-1][diff_bool_v][-1]
    sp_dur = round(sp_dur,3)
    return sp_dur


def is_calcium_spike(t,nexus_v):
    th_start_nexus_mV = -40
    th_end_nexus_mV = -42

    logical_above_start_nexus_v = nexus_v > th_start_nexus_mV
    logical_above_end_nexus_v = nexus_v > th_end_nexus_mV
    diff_logical_above_start_nexus_v = np.diff(logical_above_start_nexus_v)
    diff_logical_above_end_nexus_v = np.diff(logical_above_end_nexus_v)

    if np.any(diff_logical_above_start_nexus_v):
        bool = 1
        start_t = t[np.where(diff_logical_above_start_nexus_v)][0]
        end_t = t[:-1][np.where(diff_logical_above_end_nexus_v)][1] # this is correct if there is more than one spike
    else:
        bool = 0
        start_t = None
        end_t = None

    return (bool,start_t,end_t)


def v_ncline(Vs, param_dict):
    '''
    dv/dt=0:
    ========
    0 = ( I_ext - gCa_HVAbar*mCa*hCa*(V-ECa) - gImbar*mIm*(V-EK) - gL*(V-EL) ) / Cm
    -> gCa_HVAbar*mCa*hCa*(V-ECa) + gImbar*mIm*(V-EK) + gL*(V-EL) = I_ext
    -> gImbar*mIm*(V-EK) = I_ext - gCa_HVAbar*mCa*hCa*(V-ECa) - gL*(V-EL)
    -> mIm = ( I_ext - gCa_HVAbar*mCa*hCa*(V-ECa) - gL*(V-EL) ) / gImbar*(V-EK)
    '''

    I_ext = param_dict['I_ext_per_area']
    gCa_HVAbar = param_dict['gCa_HVAbar']
    gImbar = param_dict['gImbar']
    E_Ca = param_dict['E_Ca']
    E_l = param_dict['E_l']
    E_K = param_dict['E_K']
    gl_pas = param_dict['gl_pas']
    Ca_HVA_h_inf = param_dict['Ca_HVA_h_inf']


    Ca_HVA_m_infs = np.array([mdf.Ca_HVA_m_inf(Vs_i) for Vs_i in Vs])

    # Im_m = (I_ext_per_area - gCa_HVAbar * Ca_HVA_m_infs * Ca_HVA_h_inf * (Vs - E_Ca) - gl_pas * (Vs - E_l)) / \
    #        (gImbar * (Vs - E_K))

    Im_m = (I_ext - gCa_HVAbar * Ca_HVA_m_infs * Ca_HVA_h_inf * (Vs - E_Ca) - gl_pas * (Vs - E_l)) / \
           (gImbar * (Vs - E_K))

    return Im_m


def Im_m_ncline(Vs, param_dict):
    '''
    dIm_m/dt=0:
    ==========
    0 = (m_inf - Im_m)/m_tau
    -> Im_m = m_inf
    -> Im_m = alpha / (alpha + beta)
    -> Im_m = k * np.exp(slope* fac * (v - shift)) / \
            ( k * np.exp(slope* fac * (v - shift)) + k * np.exp(-slope* fac * (v - shift)))
    '''

    fac = 0.04
    shift = param_dict['shift']
    slope = param_dict['slope']
    k = param_dict['k']

    Im_m = k * np.exp(slope * fac * (Vs - shift)) / \
           ( k * np.exp(slope * fac * (Vs - shift) ) + k * np.exp(-slope * fac * (Vs - shift)))

    return Im_m
