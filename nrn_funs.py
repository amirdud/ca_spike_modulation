from neuron import h, gui
import random
import pickle
import copy
import numpy as np
from constants import *


def get_random_segments(sections, n_segs, seed=False):
    '''
    get a list of random segments from a list of sections (from a neuron model)

    sections: list of sections from a neuron model
    n_segs: number of segments to return from all these sections

    return: segments_subsample: a list of segments from all spiecified sections
    '''
    segments = []

    for sec in sections:
        for seg in sec:
            segments.append(seg)

    if isinstance(seed, int):
        random.seed(seed)

    segments_subsample = random.sample(segments, n_segs)

    return segments_subsample


def get_L5PC_model():
    h.load_file("import3d.hoc")
    h.load_file(PATH_L5PC_BIOPHYSICS)
    h.load_file(PATH_L5PC_TEMPLATE)

    h.define_shape()
    L5PC = h.L5PCtemplate(PATH_L5PC_MORPHOLOGY)

    return L5PC


def define_nrn_general_params(params):
    for key in params:
        if key == 'tstop':
            h.tstop = params["tstop"]
        elif key == 'v_init':
            h.v_init = params["v_init"]
        elif key == 'celsius':
            h.celsius = params["celsius"]
        elif key == 'dt':
            h.dt = params["dt"]
        else:
            print(f"{key} parameter was not assigned to h object")


def get_soma(L5PC):
    return L5PC.apic[SOMA_SEC](SOMA_RELATIVE_SEG)


def get_nexus(L5PC):
    return L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG)


def epsc_like_current_insert(seg):
    syn = h.epsp(seg)

    return syn


def epsc_like_current_add_params(syn, params):
    for key in params:
        if key == 'tau0':
            syn.tau0 = params["tau0"]
        elif key == 'tau1':
            syn.tau1 = params["tau1"]
        elif key == 'onset':
            syn.onset = params["onset"]
        elif key == 'imax':
            syn.imax = params["imax"]
        else:
            print(f"{key} parameter was not assigned to syn object")


def iclamp_current_insert(seg):
    stim = h.IClamp(seg)

    return stim


def iclamp_current_add_params(stim, params):
    stim.delay = params["delay"]
    stim.amp = params["amp"]
    stim.dur = params["dur"]


def epsc_like_current_insert_to_multi_sections(L5PC, sections):

    # make framework:
    areas_list = []
    dic_L = {}
    dic_syns_pert = {}

    for i, sec in enumerate(sections):
        dic_L[i] = sec.L

        for j, seg in enumerate(sec):

            # add epsp-like currents to segments
            syn_pert = h.epsp(seg)

            syn_pert.tau0 = 0
            syn_pert.tau1 = 0
            syn_pert.onset = 0
            syn_pert.imax = 0

            dic_syns_pert[(i, j)] = syn_pert

            # record current in nexus
            if sec == L5PC.apic[NEXUS_APICAL_SEC] and seg == L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG):
                nexus_inds = i, j
                print('found nexus!')

            # calc area
            A = seg.area()
            areas_list.append(A)

    areas_np = np.array(areas_list)
    areas_sum = areas_np.sum()

    output_dic = {}
    output_dic["dic_syns_pert"] = dic_syns_pert
    output_dic["nexus_inds"] = nexus_inds
    output_dic["areas_sum"] = areas_sum

    return output_dic


def constant_current_insert_to_multi_sections(L5PC, sections):

    # make framework:
    areas_list = []
    dic_L = {}
    dic_stims = {}

    for i, sec in enumerate(sections):
        dic_L[i] = sec.L

        for j, seg in enumerate(sec):

            # add constant current to segments
            stim = h.IClamp(seg)

            stim.amp = 0
            stim.dur = 1000
            stim.delay = 0

            dic_stims[(i, j)] = stim

            # record current in nexus
            if sec == L5PC.apic[NEXUS_APICAL_SEC] and seg == L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG):
                nexus_inds = i, j
                print('found nexus!')

            # calc area
            A = seg.area()
            areas_list.append(A)

    areas_np = np.array(areas_list)
    areas_sum = areas_np.sum()

    output_dic = {}
    output_dic["dic_stims"] = dic_stims
    output_dic["nexus_inds"] = nexus_inds
    output_dic["areas_sum"] = areas_sum

    return output_dic


def get_gca_hva_from_multi_sections(sections):

    # get gbars:
    dic_gbars = {}
    for i, sec in enumerate(sections):
        for j, seg in enumerate(sec):
            dic_gbars[(i, j)] = seg.gCa_HVAbar_Ca_HVA

    return dic_gbars


def get_nexus_model_all_channels(model_params):
    # read nexus parameters
    file = open("nexus_point_neuron_parameters.pkl", 'rb')
    nexus_parameters = pickle.load(file)
    file.close()

    soma = h.Section(name='soma')
    soma.L = model_params["L"]
    soma.diam = model_params["diam"]

    soma.insert('pas')
    soma(SOMA_RELATIVE_SEG).g_pas = nexus_parameters['g_pas']
    soma(SOMA_RELATIVE_SEG).e_pas = nexus_parameters['e_pas']

    soma.insert('CaDynamics_E2')
    soma(SOMA_RELATIVE_SEG).gamma_CaDynamics_E2 = nexus_parameters['gamma_CaDynamics_E2']
    soma(SOMA_RELATIVE_SEG).decay_CaDynamics_E2 = nexus_parameters['decay_CaDynamics_E2']

    soma.insert('Ca_HVA')
    soma(SOMA_RELATIVE_SEG).gCa_HVAbar_Ca_HVA = nexus_parameters['gCa_HVAbar_Ca_HVA']

    soma.insert('Ca_LVAst')
    soma(SOMA_RELATIVE_SEG).gCa_LVAstbar_Ca_LVAst = nexus_parameters['gCa_LVAstbar_Ca_LVAst']

    soma.insert('Ih')
    soma(SOMA_RELATIVE_SEG).gIhbar_Ih = nexus_parameters['gIhbar_Ih']

    soma.insert('Im')
    soma(SOMA_RELATIVE_SEG).gImbar_Im = nexus_parameters['gImbar_Im']

    soma.insert('NaTs2_t')
    soma(SOMA_RELATIVE_SEG).gNaTs2_tbar_NaTs2_t = nexus_parameters['gNaTs2_tbar_NaTs2_t']

    soma.insert('SK_E2')
    soma(SOMA_RELATIVE_SEG).gSK_E2bar_SK_E2 = nexus_parameters['gSK_E2bar_SK_E2']

    soma.insert('SKv3_1')
    soma(SOMA_RELATIVE_SEG).gSKv3_1bar_SKv3_1 = nexus_parameters['gSKv3_1bar_SKv3_1']

    return soma
