import numpy as np
import matplotlib.pyplot as plt
import util_funs as uf
import scipy.signal


def calc_integral(x, y, start_ind=0, stop_ind=-1, y_0=0, show=False):
    y_use = y[start_ind:stop_ind] - y_0
    x_use = x[start_ind:stop_ind]

    area = np.trapz(y_use, x_use)

    if show:
        f, ax = plt.subplots()
        ax.plot(x_use, y_use)
        ax.set_title(area)

    return area


def n_spikes(soma_v_np):
    th_sp = 0.15  # mV/ms
    diff_soma_v = np.diff(soma_v_np)
    smooth_diff_soma_v = uf.smooth(diff_soma_v, 35)
    loc_peaks = scipy.signal.find_peaks(smooth_diff_soma_v, height=th_sp)[0]
    n_sp = loc_peaks.size
    return n_sp


def time_first_spike(soma_v_np, Fs):
    th_sp = 0.15  # mV/ms
    diff_soma_v = np.diff(soma_v_np)
    smooth_diff_soma_v = uf.smooth(diff_soma_v, 35)
    loc_peaks = scipy.signal.find_peaks(smooth_diff_soma_v, height=th_sp)[0]
    timing_1st = loc_peaks[0] / Fs
    return timing_1st


def calc_spike_duration(t, v):
    mV_th = -70
    bool_v = v > mV_th
    diff_bool_v = np.diff(bool_v)
    sp_dur = t[:-1][diff_bool_v][-1]
    sp_dur = round(sp_dur, 3)
    return sp_dur


def is_calcium_spike(t, nexus_v):
    th_start_nexus_mV = -40
    th_end_nexus_mV = -42

    logical_above_start_nexus_v = nexus_v > th_start_nexus_mV
    logical_above_end_nexus_v = nexus_v > th_end_nexus_mV
    diff_logical_above_start_nexus_v = np.diff(logical_above_start_nexus_v)
    diff_logical_above_end_nexus_v = np.diff(logical_above_end_nexus_v)

    if np.any(diff_logical_above_start_nexus_v):
        bool = 1
        start_t = t[np.where(diff_logical_above_start_nexus_v)][0]
        end_t = t[:-1][np.where(diff_logical_above_end_nexus_v)][1]  # for more than one spike
    else:
        bool = 0
        start_t = None
        end_t = None

    return (bool, start_t, end_t)


def order_currents(current_list, current_names_list):

    max_vals = []
    for (current, current_name) in zip(current_list, current_names_list):

        max_sign_current = max(abs(current))
        if current_name in ["I_SK", "I_SKv3", "I_Im"]:
            max_sign_current = max_sign_current * (-1)

        max_vals.append(max_sign_current)

    max_vals_sorted_inds = np.argsort(max_vals)

    return max_vals_sorted_inds


