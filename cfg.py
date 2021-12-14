import numpy as np

# figure 1
# ========
cfg_fig_1_1 = {}

cfg_fig_1_1["h"] = {}
cfg_fig_1_1["h"]["tstop"] = 300  # ms
cfg_fig_1_1["h"]["v_init"] = -80
cfg_fig_1_1["h"]["celsius"] = 37
cfg_fig_1_1["h"]["save_figs"] = False

cfg_fig_1_1["syn_nexus"] = {}
cfg_fig_1_1["syn_nexus"]["tau0"] = 0.5
cfg_fig_1_1["syn_nexus"]["tau1"] = 5
cfg_fig_1_1["syn_nexus"]["onset"] = 155
cfg_fig_1_1["syn_nexus"]["imax"] = 1.6


cfg_fig_1_2 = {}
cfg_fig_1_2["h"] = {}
cfg_fig_1_2["h"]["tstop"] = 250  # ms
cfg_fig_1_2["h"]["v_init"] = -80
cfg_fig_1_2["h"]["celsius"] = 37
cfg_fig_1_2["h"]["save_figs"] = False
cfg_fig_1_2["h"]["run_sim"] = False
cfg_fig_1_2["h"]["bef_peak_t"] = 20  # msec
cfg_fig_1_2["h"]["aft_peak_t"] = 60
cfg_fig_1_2["h"]["t1"] = 1
cfg_fig_1_2["h"]["t2"] = 11
cfg_fig_1_2["h"]["t3"] = 21
cfg_fig_1_2["h"]["t4"] = 31

cfg_fig_1_2["stim_soma"] = {}
cfg_fig_1_2["stim_soma"]["delay"] = np.array([150])
cfg_fig_1_2["stim_soma"]["amp"] = np.round(np.arange(0, 5.1, 2), 1)
cfg_fig_1_2["stim_soma"]["dur"] = np.round(np.arange(4, 6, 1), 1)

cfg_fig_1_2["syn_nexus"] = {}
cfg_fig_1_2["syn_nexus"]["tau0"] = np.round(np.arange(0.5, 0.6, 0.1), 1)
cfg_fig_1_2["syn_nexus"]["tau1"] = np.round(np.arange(5, 6, 3), 1)
cfg_fig_1_2["syn_nexus"]["BACdt"] = np.round(np.arange(0, 5 + 2.5, 2.5), 1)
cfg_fig_1_2["syn_nexus"]["imax"] = np.round(np.arange(0, 3.1, 1), 1)


# figure 2
# =========
cfg_fig_2_1 = {}
cfg_fig_2_1["h"] = {}
cfg_fig_2_1["h"]["tstop"] = 250  # ms
cfg_fig_2_1["h"]["v_init"] = -80
cfg_fig_2_1["h"]["celsius"] = 37
cfg_fig_2_1["h"]["plot_condition"] = 'other'
# cfg_fig_2_1["h"]["plot_condition"] = 'strong_inh'
cfg_fig_2_1["h"]["save_figs"] = False

cfg_fig_2_1["syn_gen_nexus"] = {}
cfg_fig_2_1["syn_gen_nexus"]["tau0"] = 0.5
cfg_fig_2_1["syn_gen_nexus"]["tau1"] = 5
cfg_fig_2_1["syn_gen_nexus"]["onset"] = 150
cfg_fig_2_1["syn_gen_nexus"]["imax"] = 1.6

cfg_fig_2_1["syn_pert_nexus"] = {}
cfg_fig_2_1["syn_pert_nexus"]["tau0"] = np.array([0.5])
cfg_fig_2_1["syn_pert_nexus"]["tau1"] = np.array([5])
cfg_fig_2_1["syn_pert_nexus"]["onset"] = np.array([15])
cfg_fig_2_1["syn_pert_nexus"]["imax"] = np.arange(-6, 6 + 0.5, 0.5)

cfg_fig_2_2 = {}
cfg_fig_2_2["h"] = {}
cfg_fig_2_2["h"]["tstop"] = 250  # ms
cfg_fig_2_2["h"]["v_init"] = -80
cfg_fig_2_2["h"]["celsius"] = 37
cfg_fig_2_2["h"]["save_figs"] = False

cfg_fig_2_2["syn_gen_nexus"] = {}
cfg_fig_2_2["syn_gen_nexus"]["tau0"] = 0.5
cfg_fig_2_2["syn_gen_nexus"]["tau1"] = 5
cfg_fig_2_2["syn_gen_nexus"]["onset"] = 150
cfg_fig_2_2["syn_gen_nexus"]["imax"] = 1.6

cfg_fig_2_2["syn_pert_nexus"] = {}
cfg_fig_2_2["syn_pert_nexus"]["tau0"] = np.array([0.5])
cfg_fig_2_2["syn_pert_nexus"]["tau1"] = np.array([5])
cfg_fig_2_2["syn_pert_nexus"]["onset"] = np.round(np.arange(5, 35, 2.5), 2)
cfg_fig_2_2["syn_pert_nexus"]["imax"] = np.array([0, 4])


cfg_fig_2_3 = {}
cfg_fig_2_3["h"] = {}
cfg_fig_2_3["h"]["tstop"] = 250  # ms
cfg_fig_2_3["h"]["v_init"] = -80
cfg_fig_2_3["h"]["celsius"] = 37
cfg_fig_2_3["h"]["run_sim"] = False
cfg_fig_2_3["h"]["save_figs"] = False

cfg_fig_2_3["syn_gen_nexus"] = {}
cfg_fig_2_3["syn_gen_nexus"]["tau0"] = 0.5
cfg_fig_2_3["syn_gen_nexus"]["tau1"] = 5
cfg_fig_2_3["syn_gen_nexus"]["onset"] = 150
cfg_fig_2_3["syn_gen_nexus"]["imax"] = 1.6

cfg_fig_2_3["syn_pert_nexus"] = {}
cfg_fig_2_3["syn_pert_nexus"]["tau0"] = np.array([0.5])
cfg_fig_2_3["syn_pert_nexus"]["tau1"] = np.array([5])
cfg_fig_2_3["syn_pert_nexus"]["onset"] = np.round(np.arange(5, 25 + 0.5, 0.5), 2)
cfg_fig_2_3["syn_pert_nexus"]["imax"] = np.round(np.arange(-1, 1 + 0.05, 0.05), 2)


cfg_fig_2_4 = {}
cfg_fig_2_4["h"] = {}
cfg_fig_2_4["h"]["tstop"] = 250  # ms
cfg_fig_2_4["h"]["v_init"] = -80
cfg_fig_2_4["h"]["celsius"] = 37
cfg_fig_2_4["h"]["dt"] = 0.025
cfg_fig_2_4["h"]["bef_peak_t"] = 5
cfg_fig_2_4["h"]["save_figs"] = False

cfg_fig_2_4["stim_constant_nexus"] = {}
cfg_fig_2_4["stim_constant_nexus"]["amp"] = np.round(np.arange(0.8, 2.6 + 0.2, 0.2), 2)
cfg_fig_2_4["stim_constant_nexus"]["dur"] = 1000
cfg_fig_2_4["stim_constant_nexus"]["delay"] = 0


cfg_fig_2_5 = {}
cfg_fig_2_5["h"] = {}
cfg_fig_2_5["h"]["tstop"] = 250  # ms
cfg_fig_2_5["h"]["v_init"] = -80
cfg_fig_2_5["h"]["celsius"] = 37
cfg_fig_2_5["h"]["dt"] = 0.025
cfg_fig_2_5["h"]["save_figs"] = False

cfg_fig_2_5["syn_gen_nexus"] = {}
cfg_fig_2_5["syn_gen_nexus"]["tau0"] = 0.5
cfg_fig_2_5["syn_gen_nexus"]["tau1"] = 5
cfg_fig_2_5["syn_gen_nexus"]["onset"] = 150
cfg_fig_2_5["syn_gen_nexus"]["imax"] = 1.6

cfg_fig_2_5["modulation"] = {}
cfg_fig_2_5["modulation"]["gCa_HVA_muls"] = np.round(np.arange(1, 1.2 + 0.02, 0.02), 2)


cfg_fig_2_6 = {}
cfg_fig_2_6["h"] = {}
cfg_fig_2_6["h"]["tstop"] = 250  # ms
cfg_fig_2_6["h"]["v_init"] = -80
cfg_fig_2_6["h"]["celsius"] = 37
cfg_fig_2_6["h"]["dt"] = 0.025
cfg_fig_2_6["h"]["save_figs"] = False
cfg_fig_2_6["h"]["factor"] = 100
cfg_fig_2_6["h"]["plot_condition"] = 'epsp_and_weak_ipsp'
# cfg_fig_2_6["h"]["plot_condition"] = 'strong_ipsp'

cfg_fig_2_6["syn_gen_nexus"] = {}
cfg_fig_2_6["syn_gen_nexus"]["tau0"] = 0.5
cfg_fig_2_6["syn_gen_nexus"]["tau1"] = 5
cfg_fig_2_6["syn_gen_nexus"]["onset"] = 150
cfg_fig_2_6["syn_gen_nexus"]["imax"] = 1.6

cfg_fig_2_6["syn_pert_nexus"] = {}
cfg_fig_2_6["syn_pert_nexus"]["tau0"] = 0.5
cfg_fig_2_6["syn_pert_nexus"]["tau1"] = 1.2
cfg_fig_2_6["syn_pert_nexus"]["onset"] = 15
cfg_fig_2_6["syn_pert_nexus"]["imax_strong_ipsp"] = np.arange(-5, -4 + 0.25, 0.25)
cfg_fig_2_6["syn_pert_nexus"]["imax_weak_ipsp"] = np.arange(-1.75, 0, 0.25)
cfg_fig_2_6["syn_pert_nexus"]["imax_epsp"] = np.arange(0, 2 + 0.25, 0.25)

# figure 3
# =========
cfg_fig_3_1 = {}
cfg_fig_3_1["h"] = {}
cfg_fig_3_1["h"]["tstop"] = 250  # ms
cfg_fig_3_1["h"]["v_init"] = -80
cfg_fig_3_1["h"]["celsius"] = 37
cfg_fig_3_1["h"]["save_figs"] = False

cfg_fig_3_1["syn_gen_nexus"] = {}
cfg_fig_3_1["syn_gen_nexus"]["tau0"] = 0.5
cfg_fig_3_1["syn_gen_nexus"]["tau1"] = 5
cfg_fig_3_1["syn_gen_nexus"]["onset"] = 155
cfg_fig_3_1["syn_gen_nexus"]["imax"] = 1.6


cfg_fig_3_2 = {}
cfg_fig_3_2["h"] = {}
cfg_fig_3_2["h"]["tstop"] = 250  # ms
cfg_fig_3_2["h"]["v_init"] = -80
cfg_fig_3_2["h"]["celsius"] = 37
cfg_fig_3_2["h"]["save_figs"] = False

cfg_fig_3_2["syn_gen"] = {}
cfg_fig_3_2["syn_gen"]["tau0"] = 0.5
cfg_fig_3_2["syn_gen"]["tau1"] = 5
cfg_fig_3_2["syn_gen"]["onset"] = 150
cfg_fig_3_2["syn_gen"]["imax"] = 0.03

cfg_fig_3_2["model"] = {}
cfg_fig_3_2["model"]["L"] = 10
cfg_fig_3_2["model"]["diam"] = 10


# figure 4
# =========
cfg_fig_4_1 = {}
cfg_fig_4_1["h"] = {}
cfg_fig_4_1["h"]["v"] = np.arange(-100, 150, 1).astype(np.float32)
cfg_fig_4_1["h"]["save_figs"] = False

cfg_fig_4_1["Im"] = {}
cfg_fig_4_1["Im"]["shift"] = -39
cfg_fig_4_1["Im"]["slope"] = 2.3
cfg_fig_4_1["Im"]["k"] = 2.0e-3


cfg_fig_4_2 = {}
cfg_fig_4_2["h"] = {}
cfg_fig_4_2["h"]["tmin"] = 0
cfg_fig_4_2["h"]["tmax"] = 150
cfg_fig_4_2["h"]["dt"] = 0.025
cfg_fig_4_2["h"]["save_figs"] = False

cfg_fig_4_2["phys"] = {}
cfg_fig_4_2["phys"]["E_K"] = -77.0
cfg_fig_4_2["phys"]["E_Ca"] = 132
cfg_fig_4_2["phys"]["E_l"] = -75
cfg_fig_4_2["phys"]["Cm"] = 1.0
cfg_fig_4_2["phys"]["A"] = 1e-5  # cm2 (18 um cell diameter)

cfg_fig_4_2["sim"] = {}
cfg_fig_4_2["sim"]["amps"] = [0]
cfg_fig_4_2["sim"]["mul_Cas"] = [5]
cfg_fig_4_2["sim"]["mul_Ks"] = [7.5]
cfg_fig_4_2["sim"]["ts"] = 10
cfg_fig_4_2["sim"]["tau1"] = 0.5
cfg_fig_4_2["sim"]["tau2"] = 5
cfg_fig_4_2["sim"]["amp_syn"] = 0.0004  # uA
cfg_fig_4_2["sim"]["stim_delay"] = 10


cfg_fig_4_3 = {}
cfg_fig_4_3["h"] = {}
cfg_fig_4_3["h"]["tmin"] = 0
cfg_fig_4_3["h"]["tmax"] = 150
cfg_fig_4_3["h"]["dt"] = 0.025
cfg_fig_4_3["h"]["save_figs"] = False

cfg_fig_4_3["phys"] = {}
cfg_fig_4_3["phys"]["E_K"] = -85.0
cfg_fig_4_3["phys"]["E_Ca"] = 120
cfg_fig_4_3["phys"]["E_l"] = -77
cfg_fig_4_3["phys"]["Cm"] = 1.0
cfg_fig_4_3["phys"]["A"] = 1e-5  # cm2 (18 um cell diameter)
cfg_fig_4_3["phys"]["shift"] = -39
cfg_fig_4_3["phys"]["slope"] = 2.3
cfg_fig_4_3["phys"]["k"] = 2.0e-3

cfg_fig_4_3["sim"] = {}
cfg_fig_4_3["sim"]["amps"] = [0]
cfg_fig_4_3["sim"]["mul_Cas"] = [5.1]
cfg_fig_4_3["sim"]["mul_Ks"] = [7.5]
cfg_fig_4_3["sim"]["mul_gpas"] = 1
cfg_fig_4_3["sim"]["ts"] = 10
cfg_fig_4_3["sim"]["tau1"] = 0.5
cfg_fig_4_3["sim"]["tau2"] = 5
cfg_fig_4_3["sim"]["amp_syn"] = 0.0003
cfg_fig_4_3["sim"]["stim_delay"] = 10
cfg_fig_4_3["sim"]["V_inits"] = np.array([-75])
cfg_fig_4_3["sim"]["Im_m_inits"] = np.array([0])

# Figure 5
# ========
cfg_fig_5_1 = {}
cfg_fig_5_1["h"] = {}
cfg_fig_5_1["h"]["tmin"] = 0
cfg_fig_5_1["h"]["tmax"] = 250
cfg_fig_5_1["h"]["dt"] = 0.025
cfg_fig_5_1["h"]["save_figs"] = False
cfg_fig_5_1["h"]["pert_type"] = 'epsp_and_weak_ipsp'
# cfg_fig_5_1["h"]["pert_type"] = 'strong_ipsp'
# cfg_fig_5_1["h"]["pert_type"] = 'epsp_delta_t'
# cfg_fig_5_1["h"]["pert_type"] = 'ACh'
# cfg_fig_5_1["h"]["pert_type"] = 'constant_current'
cfg_fig_5_1["h"]["t_plat"] = 36.8 / 2

cfg_fig_5_1["phys"] = {}
cfg_fig_5_1["phys"]["E_K"] = -85.0
cfg_fig_5_1["phys"]["E_Ca"] = 120
cfg_fig_5_1["phys"]["E_l"] = -77
cfg_fig_5_1["phys"]["Cm"] = 1.0
cfg_fig_5_1["phys"]["A"] = 1e-5  # cm2 (18 um cell diameter)
cfg_fig_5_1["phys"]["shift"] = -39
cfg_fig_5_1["phys"]["slope"] = 2.3
cfg_fig_5_1["phys"]["k"] = 2.0e-3

cfg_fig_5_1["sim"] = {}
cfg_fig_5_1["sim"]["mul_K"] = 7.5
cfg_fig_5_1["sim"]["mul_gpas"] = 1
cfg_fig_5_1["sim"]["ts_gen"] = 150
cfg_fig_5_1["sim"]["tau1_gen"] = 0.5
cfg_fig_5_1["sim"]["tau2_gen"] = 5
cfg_fig_5_1["sim"]["stim_delay"] = cfg_fig_5_1["sim"]["ts_gen"]
cfg_fig_5_1["sim"]["tau1_pert"] = 0.5
cfg_fig_5_1["sim"]["tau2_pert"] = 5
cfg_fig_5_1["sim"]["V_init"] = -75
cfg_fig_5_1["sim"]["Im_m_init"] = 0
cfg_fig_5_1["sim"]["pert_dependent_params"] = {'epsp_and_weak_ipsp':
                                                   {'amp': 0,
                                                    'mul_Ca': 5.1,
                                                    'amp_syn_gen': 0.0003,
                                                    'ts_pert': 165,
                                                    'amp_syn_perts_epsp': np.round(np.arange(0, 0.0005 + 0.00005, 0.00005), 5),  # uA
                                                    'amp_syn_perts_weak_ipsp': np.round(np.arange(-0.0003, 0, 0.00005), 5),
                                                    },
                                               'strong_ipsp':
                                                   {'amp': 0,
                                                    'mul_Ca': 5.1,
                                                    'amp_syn_gen': 0.0003,
                                                    'ts_pert': 165,
                                                    'amp_syn_perts_strong_ipsp': np.round(np.arange(-0.0009, -0.00076, 0.00002), 5),
                                                    },
                                               'epsp_delta_t':
                                                   {'amp': 0,
                                                    'mul_Ca': 5.1,
                                                    'amp_syn_gen': 0.0003,
                                                    'ts_pert': np.arange(160, 180, 2),
                                                    'amp_syn_pert': 0.0005  # uA
                                                    },
                                               'ACh':
                                                   {'amp': 0,
                                                    'mul_Ca': np.arange(5.1, 5.61, 0.05),
                                                    'amp_syn_gen': 0.0003,
                                                    'ts_pert': 165,
                                                    'amp_syn_pert': 0
                                                    },
                                               'constant_current':
                                                   {'amp': np.round(np.arange(0.00002, 0.00012, 0.00001), 5),
                                                    'mul_Ca': 5.1,
                                                    'amp_syn_gen': 0,
                                                    'ts_pert': 165,
                                                    'amp_syn_pert': 0
                                                    },
                                               }

# Figure 6
# ========
cfg_fig_6_1 = {}
cfg_fig_6_1["h"] = {}
cfg_fig_6_1["h"]["tmin"] = 0
cfg_fig_6_1["h"]["tmax"] = 120
cfg_fig_6_1["h"]["dt"] = 0.025
cfg_fig_6_1["h"]["save_figs"] = False

cfg_fig_6_1["phys"] = {}
cfg_fig_6_1["phys"]["E_K"] = -85.0
cfg_fig_6_1["phys"]["E_Ca"] = 120
cfg_fig_6_1["phys"]["E_l"] = -77
cfg_fig_6_1["phys"]["Cm"] = 1.0
cfg_fig_6_1["phys"]["A"] = 1e-5  # cm2 (18 um cell diameter)
cfg_fig_6_1["phys"]["shift"] = -39
cfg_fig_6_1["phys"]["slope"] = 2.3
cfg_fig_6_1["phys"]["k"] = 2.0e-3

cfg_fig_6_1["sim"] = {}
cfg_fig_6_1["sim"]["amp"] = 0
cfg_fig_6_1["sim"]["mul_Ca"] = 5.1
cfg_fig_6_1["sim"]["mul_K"] = 7.5
cfg_fig_6_1["sim"]["mul_gpas"] = 1
cfg_fig_6_1["sim"]["V_init"] = -30
cfg_fig_6_1["sim"]["Im_m_init"] = 0

cfg_fig_6_1["nclines"] = {}
cfg_fig_6_1["nclines"]["Im_ms_low"] = 0
cfg_fig_6_1["nclines"]["Im_ms_high"] = 1
cfg_fig_6_1["nclines"]["Im_ms_delta"] = 0.0001
cfg_fig_6_1["nclines"]["Vs_low"] = -150
cfg_fig_6_1["nclines"]["Vs_high"] = 100
cfg_fig_6_1["nclines"]["Vs_delta"] = 0.025
cfg_fig_6_1["nclines"]["n_samples"] = 35
cfg_fig_6_1["nclines"]["scale"] = 10000
cfg_fig_6_1["nclines"]["dd"] = 1
cfg_fig_6_1["nclines"]["ss"] = 1
cfg_fig_6_1["nclines"]["lw"] = 2


# Figure 7
# ========
cfg_fig_7_1 = {}
cfg_fig_7_1["h"] = {}
cfg_fig_7_1["h"]["tmin_pre"] = 0.0
cfg_fig_7_1["h"]["tmax_post"] = 100.0
# cfg_fig_7_1["h"]["dt"] = 0.0025
cfg_fig_7_1["h"]["dt"] = 0.05
cfg_fig_7_1["h"]["save_figs"] = False
cfg_fig_7_1["h"]["t_plat"] = 11.7 / 2

cfg_fig_7_1["phys"] = {}
cfg_fig_7_1["phys"]["E_K"] = -85.0
cfg_fig_7_1["phys"]["E_Ca"] = 120
cfg_fig_7_1["phys"]["E_l"] = -77
cfg_fig_7_1["phys"]["Cm"] = 1.0
cfg_fig_7_1["phys"]["A"] = 1e-5  # cm2 (18 um cell diameter)
cfg_fig_7_1["phys"]["shift"] = -39
cfg_fig_7_1["phys"]["slope"] = 2.3
cfg_fig_7_1["phys"]["k"] = 2.0e-3
cfg_fig_7_1["phys"]["th_end_nexus_mV"] = -75

cfg_fig_7_1["sim"] = {}
cfg_fig_7_1["sim"]["amp"] = 0
cfg_fig_7_1["sim"]["mul_Ca"] = 5.1
cfg_fig_7_1["sim"]["mul_K"] = 7.5
cfg_fig_7_1["sim"]["mul_gpas"] = 1

cfg_fig_7_1["nclines"] = {}
cfg_fig_7_1["nclines"]["Im_ms_low"] = 0
cfg_fig_7_1["nclines"]["Im_ms_high"] = 1
cfg_fig_7_1["nclines"]["Im_ms_delta"] = 0.0001
cfg_fig_7_1["nclines"]["Vs_low"] = -150
cfg_fig_7_1["nclines"]["Vs_high"] = 100
cfg_fig_7_1["nclines"]["Vs_delta"] = 0.025
cfg_fig_7_1["nclines"]["n_samples"] = 35
cfg_fig_7_1["nclines"]["scale"] = 10000
cfg_fig_7_1["nclines"]["dd"] = 1
cfg_fig_7_1["nclines"]["ss"] = 1
cfg_fig_7_1["nclines"]["lw"] = 2
cfg_fig_7_1["nclines"]["dur_show_pert"] = 1  # ms

cfg_fig_7_1["nclines_zoom"] = {}
cfg_fig_7_1["nclines_zoom"]["scale"] = 5000
cfg_fig_7_1["nclines_zoom"]["marks"] = 4
cfg_fig_7_1["nclines_zoom"]["markew"] = 0.1
cfg_fig_7_1["nclines_zoom"]["npoints"] = 40  # ms


# Figure 8
# ========
cfg_fig_8 = {}
cfg_fig_8["h"] = {}
cfg_fig_8["h"]["tmin"] = 0.0
cfg_fig_8["h"]["tmax"] = 300.0
cfg_fig_8["h"]["dt"] = 0.0025
cfg_fig_8["h"]["save_figs"] = False

cfg_fig_8["phys"] = {}
cfg_fig_8["phys"]["E_K"] = -85.0
cfg_fig_8["phys"]["E_Ca"] = 120
cfg_fig_8["phys"]["E_l"] = -77
cfg_fig_8["phys"]["Cm"] = 1.0
cfg_fig_8["phys"]["A"] = 1e-5  # cm2 (18 um cell diameter)
cfg_fig_8["phys"]["shift"] = -39
cfg_fig_8["phys"]["slope"] = 2.3
cfg_fig_8["phys"]["k"] = 2.0e-3
cfg_fig_8["phys"]["th_end_nexus_mV"] = -75

cfg_fig_8["sim"] = {}
cfg_fig_8["sim"]["mul_K"] = 7.5
cfg_fig_8["sim"]["mul_gpas"] = 1
cfg_fig_8["sim"]["V_init"] = -30

cfg_fig_8["nclines"] = {}
cfg_fig_8["nclines"]["Im_ms_low"] = 0
cfg_fig_8["nclines"]["Im_ms_high"] = 1
cfg_fig_8["nclines"]["Im_ms_delta"] = 0.0001
cfg_fig_8["nclines"]["Vs_low"] = -150
cfg_fig_8["nclines"]["Vs_high"] = 100
cfg_fig_8["nclines"]["Vs_delta"] = 0.025
cfg_fig_8["nclines"]["n_samples"] = 35
cfg_fig_8["nclines"]["scale"] = 10000
cfg_fig_8["nclines"]["dd"] = 1
cfg_fig_8["nclines"]["ss"] = 1
cfg_fig_8["nclines"]["lw"] = 2

cfg_fig_8_1 = {}
cfg_fig_8_2 = {}
cfg_fig_8_1["sim"] = {}
cfg_fig_8_2["sim"] = {}
cfg_fig_8_1["sim"]["amp"] = 0
cfg_fig_8_2["sim"]["amp"] = np.array([0, 0.00015, 0.0003])
cfg_fig_8_1["sim"]["mul_Ca"] = np.array([5.1, 6.1, 7.1])
cfg_fig_8_2["sim"]["mul_Ca"] = 5.1
cfg_fig_8_1["sim"]["Im_m_init"] = 0
cfg_fig_8_2["sim"]["Im_m_init"] = np.array([0, 0.0683, 0.1287])  # for better phase plane visualization

cfg_fig_8_3 = {}
cfg_fig_8_3["h"] = {}
cfg_fig_8_3["h"]["tmin"] = 0.0
cfg_fig_8_3["h"]["tmax"] = 500.0
cfg_fig_8_3["h"]["dt"] = 0.025
cfg_fig_8_3["h"]["save_figs"] = False
cfg_fig_8_3["h"]["pert_type"] = 'ACh'
# cfg_fig_8_3["h"]["pert_type"] = 'constant_current'
# cfg_fig_8_3["h"]["pert_type"] = 'ACh_K'

cfg_fig_8_3["sim"] = {}
cfg_fig_8_3["sim"]["Im_m_init"] = 0
cfg_fig_8_3["sim"]["pert_dependent_params"] = {'ACh':
                                                   {'mul_Cas': np.round(np.arange(5.1, 9 + 0.1, 0.1), 2),
                                                    'mul_Ks': np.array([7.5]),
                                                    'amps': np.array([0]),
                                                    },
                                               'constant_current':
                                                   {'amps': np.round(np.arange(0, 0.00034 + 0.00001, 0.00001), 5),
                                                    'mul_Cas': [5.1],
                                                    'mul_Ks': np.array([7.5]),
                                                    },
                                               'ACh_K':
                                                   {'mul_Ks': np.round(np.arange(7.5, 4 - 0.1, -0.1), 2),
                                                    'mul_Cas': [5.1],
                                                    'amps': np.array([0]),
                                                    },
                                               }

# Figure 9
# ========
cfg_fig_9 = {}
cfg_fig_9["h"] = {}
cfg_fig_9["h"]["tmin"] = 0.0
cfg_fig_9["h"]["tmax"] = 200.0
cfg_fig_9["h"]["dt"] = 0.025
cfg_fig_9["h"]["save_figs"] = False
cfg_fig_9["h"]["pert_type"] = 'ctrl'
# cfg_fig_9["h"]["pert_type"] = 'ipsp'
# cfg_fig_9["h"]["pert_type"] = 'epsp'
# cfg_fig_9["h"]["pert_type"] = 'ACh'

# parameters from:
# http://www.math.pitt.edu/~bdoiron/assets/ermentrout-and-terman-ch-1.pdf
cfg_fig_9["phys"] = {}
cfg_fig_9["phys"]["shift"] = -39
cfg_fig_9["phys"]["slope"] = 2.3
cfg_fig_9["phys"]["k"] = 2.0e-3
cfg_fig_9["phys"]["gK"] = 36
cfg_fig_9["phys"]["gNa"] = 120
cfg_fig_9["phys"]["gL"] = 0.3
cfg_fig_9["phys"]["g_con"] = 0.4
cfg_fig_9["phys"]["A_soma"] = 1e-5
cfg_fig_9["phys"]["A_nexus"] = 1e-5
cfg_fig_9["phys"]["Cm_soma"] = 1.0
cfg_fig_9["phys"]["Cm_nexus"] = 1.0
cfg_fig_9["phys"]["E_K"] = -77
cfg_fig_9["phys"]["E_Na"] = 50
cfg_fig_9["phys"]["E_l"] = -54.4
cfg_fig_9["phys"]["E_Ca"] = 120

cfg_fig_9["sim"] = {}
cfg_fig_9["sim"]["amp_soma"] = 0
cfg_fig_9["sim"]["amp_nexus"] = 0
cfg_fig_9["sim"]["Im_m_init"] = 0
cfg_fig_9["sim"]["mul_gpas"] = 1
cfg_fig_9["sim"]["mul_K"] = 7.5
cfg_fig_9["sim"]["amp_nexus_syn_gen"] = 0.0004
cfg_fig_9["sim"]["ts_nexus_gen"] = 50
cfg_fig_9["sim"]["tau1_nexus_gen"] = 0.5
cfg_fig_9["sim"]["tau2_nexus_gen"] = 8
cfg_fig_9["sim"]["tau1_nexus_pert"] = 0.5
cfg_fig_9["sim"]["tau2_nexus_pert"] = 5
cfg_fig_9["sim"]["V_soma_init"] = -65.0
cfg_fig_9["sim"]["V_nexus_init"] = -65.0
cfg_fig_9["sim"]["pert_dependent_params"] = {'ctrl':
                                                 {'amp_nexus_syn_pert': 0,
                                                  'mul_Ca': 7,
                                                  },
                                             'ipsp':
                                                 {'amp_nexus_syn_pert': -0.0004,
                                                  'mul_Ca': 7,
                                                  },
                                             'epsp':
                                                 {'amp_nexus_syn_pert': 0.0012,
                                                  'mul_Ca': 7,
                                                  },
                                             'ACh':
                                                 {'amp_nexus_syn_pert': 0,
                                                  'mul_Ca': 7.5,
                                                  },
                                             }

cfg_fig_9_1 = {}
cfg_fig_9_1["sim"] = {}
cfg_fig_9_1["sim"]["ts_nexus_pert"] = 58

cfg_fig_9_2 = {}
cfg_fig_9_2["sim"] = {}
cfg_fig_9_2["sim"]["ts_nexus_pert"] = np.arange(52, 80, 2)
cfg_fig_9_2["sim"]["amp_nexus_syn_pert"] = np.round(np.arange(-0.00175, 0.002, 0.00025), 5)
cfg_fig_9_2["sim"]["mul_Ca"] = np.array([7])

cfg_fig_9_3 = {}
cfg_fig_9_3["sim"] = {}
cfg_fig_9_3["sim"]["ts_nexus_pert"] = 58
cfg_fig_9_3["sim"]["amp_nexus_syn_pert"] = 0
cfg_fig_9_3["sim"]["mul_Ca"] = np.round(np.arange(7, 9.1, 0.1), 1)
