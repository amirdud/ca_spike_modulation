from neuron import h, gui
from constants import *



def L5PC_define_vars(stims=None):
    # stims: dict of {<stim_name_1>: <stim_1>,
    #                 <stim_name_2>: < stim_2>, ...}

    var = {}
    var['t'] = h.Vector()

    var["nexus"] = {}
    var["nexus"]["v"] = h.Vector()
    var["nexus"]["I_CaHVA"] = h.Vector()
    var["nexus"]["I_CaLVA"] = h.Vector()
    var["nexus"]["I_SK"] = h.Vector()
    var["nexus"]["I_SKv3"] = h.Vector()
    var["nexus"]["I_Im"] = h.Vector()
    var["nexus"]["I_Ih"] = h.Vector()
    var["nexus"]["I_NaTs2_t"] = h.Vector()
    var["nexus"]["I_K"] = h.Vector()
    var["nexus"]["I_Na"] = h.Vector()
    var["nexus"]["I_Ca"] = h.Vector()

    var["nexus"]["g_CaHVA"] = h.Vector()
    var["nexus"]["g_CaLVA"] = h.Vector()
    var["nexus"]["g_SK"] = h.Vector()
    var["nexus"]["g_SKv3"] = h.Vector()
    var["nexus"]["g_Im"] = h.Vector()
    var["nexus"]["g_Ih"] = h.Vector()
    var["nexus"]["g_NaTs2_t"] = h.Vector()

    var["nexus"]["m_CaHVA"] = h.Vector()
    var["nexus"]["h_CaHVA"] = h.Vector()
    var["nexus"]["m_CaLVA"] = h.Vector()
    var["nexus"]["h_CaLVA"] = h.Vector()
    var["nexus"]["z_SK"] = h.Vector()
    var["nexus"]["m_SKv3"] = h.Vector()
    var["nexus"]["m_Im"] = h.Vector()
    var["nexus"]["m_Ih"] = h.Vector()
    var["nexus"]["m_NaTs2_t"] = h.Vector()
    var["nexus"]["h_NaTs2_t"] = h.Vector()

    var["soma"] = {}
    var["soma"]["v"] = h.Vector()
    var["soma"]["I_K"] = h.Vector()
    var["soma"]["I_Na"] = h.Vector()
    var["soma"]["I_Ca"] = h.Vector()

    if stims:
        var["stims"] = {}
        for key in stims:
            var["stims"]['I_' + key] = h.Vector()

    return var


def L5PC_record(var, L5PC, stims=None):
    var['t'].record(h._ref_t)

    var['nexus']["v"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG)._ref_v)
    var["nexus"]['I_CaHVA'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_HVA._ref_ica)
    var["nexus"]['I_CaLVA'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_LVAst._ref_ica)
    var["nexus"]['I_SK'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).SK_E2._ref_ik)
    var["nexus"]['I_SKv3'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).SKv3_1._ref_ik)
    var["nexus"]['I_Im'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Im._ref_ik)
    var["nexus"]['I_Ih'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ih._ref_ihcn)
    var["nexus"]['I_NaTs2_t'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).NaTs2_t._ref_ina)
    var["nexus"]['I_K'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).k_ion._ref_ik)
    var["nexus"]['I_Na'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).na_ion._ref_ina)
    var["nexus"]['I_Ca'].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).ca_ion._ref_ica)

    var["nexus"]["g_CaHVA"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_HVA._ref_gCa_HVA)
    var["nexus"]["g_CaLVA"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_LVAst._ref_gCa_LVAst)
    var["nexus"]["g_SK"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).SK_E2._ref_gSK_E2)
    var["nexus"]["g_SKv3"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).SKv3_1._ref_gSKv3_1)
    var["nexus"]["g_Im"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Im._ref_gIm)
    var["nexus"]["g_Ih"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ih._ref_gIh)
    var["nexus"]["g_NaTs2_t"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).NaTs2_t._ref_gNaTs2_t)

    var["nexus"]["m_CaHVA"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_HVA._ref_m)
    var["nexus"]["h_CaHVA"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_HVA._ref_h)
    var["nexus"]["m_CaLVA"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_LVAst._ref_m)
    var["nexus"]["h_CaLVA"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ca_LVAst._ref_h)
    var["nexus"]["z_SK"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).SK_E2._ref_z)
    var["nexus"]["m_SKv3"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).SKv3_1._ref_m)
    var["nexus"]["m_Im"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Im._ref_m)
    var["nexus"]["m_Ih"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).Ih._ref_m)
    var["nexus"]["m_NaTs2_t"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).NaTs2_t._ref_m)
    var["nexus"]["h_NaTs2_t"].record(L5PC.apic[NEXUS_APICAL_SEC](NEXUS_RELATIVE_SEG).NaTs2_t._ref_h)

    var['soma']["v"].record(L5PC.soma[SOMA_SEC](SOMA_RELATIVE_SEG)._ref_v)
    var["soma"]['I_K'].record(L5PC.soma[SOMA_SEC](SOMA_RELATIVE_SEG).k_ion._ref_ik)
    var["soma"]['I_Na'].record(L5PC.soma[SOMA_SEC](SOMA_RELATIVE_SEG).na_ion._ref_ina)
    var["soma"]['I_Ca'].record(L5PC.soma[SOMA_SEC](SOMA_RELATIVE_SEG).ca_ion._ref_ica)

    if stims:
        for key in stims:
            if (key == 'syn_nexus') or (key == 'stim_soma') or (key == 'syn_gen_nexus') \
                    or (key == 'syn_pert_nexus') or (key == 'stim_constant_nexus'):
                var['stims']['I_' + key].record(stims[key]._ref_i)

    return var


def nexus_model_all_channels_define_vars(stims=None):
    # stims: dict of {<stim_name_1>: <stim_1>,
    #                 <stim_name_2>: < stim_2>, ...}

    var = {}
    var['t'] = h.Vector()

    var["soma"] = {}
    var["soma"]["v"] = h.Vector()
    var["soma"]["I_CaHVA"] = h.Vector()
    var["soma"]["I_CaLVA"] = h.Vector()
    var["soma"]["I_SK"] = h.Vector()
    var["soma"]["I_SKv3"] = h.Vector()
    var["soma"]["I_Im"] = h.Vector()
    var["soma"]["I_Ih"] = h.Vector()
    var["soma"]["I_NaTs2_t"] = h.Vector()
    var["soma"]["I_K"] = h.Vector()
    var["soma"]["I_Na"] = h.Vector()
    var["soma"]["I_Ca"] = h.Vector()

    var["soma"]["g_CaHVA"] = h.Vector()
    var["soma"]["g_CaLVA"] = h.Vector()
    var["soma"]["g_SK"] = h.Vector()
    var["soma"]["g_SKv3"] = h.Vector()
    var["soma"]["g_Im"] = h.Vector()
    var["soma"]["g_Ih"] = h.Vector()
    var["soma"]["g_NaTs2_t"] = h.Vector()

    var["soma"]["m_CaHVA"] = h.Vector()
    var["soma"]["h_CaHVA"] = h.Vector()
    var["soma"]["m_CaLVA"] = h.Vector()
    var["soma"]["h_CaLVA"] = h.Vector()
    var["soma"]["z_SK"] = h.Vector()
    var["soma"]["m_SKv3"] = h.Vector()
    var["soma"]["m_Im"] = h.Vector()
    var["soma"]["m_Ih"] = h.Vector()
    var["soma"]["m_NaTs2_t"] = h.Vector()
    var["soma"]["h_NaTs2_t"] = h.Vector()

    if stims:
        var["stims"] = {}
        for key in stims:
            var["stims"]['I_' + key] = h.Vector()

    return var


def nexus_model_all_channels_record(var, soma, stims=None):
    var['t'].record(h._ref_t)

    var['soma']["v"].record(soma(SOMA_RELATIVE_SEG)._ref_v)
    var["soma"]['I_CaHVA'].record(soma(SOMA_RELATIVE_SEG).Ca_HVA._ref_ica)
    var["soma"]['I_CaLVA'].record(soma(SOMA_RELATIVE_SEG).Ca_LVAst._ref_ica)
    var["soma"]['I_SK'].record(soma(SOMA_RELATIVE_SEG).SK_E2._ref_ik)
    var["soma"]['I_SKv3'].record(soma(SOMA_RELATIVE_SEG).SKv3_1._ref_ik)
    var["soma"]['I_Im'].record(soma(SOMA_RELATIVE_SEG).Im._ref_ik)
    var["soma"]['I_Ih'].record(soma(SOMA_RELATIVE_SEG).Ih._ref_ihcn)
    var["soma"]['I_NaTs2_t'].record(soma(SOMA_RELATIVE_SEG).NaTs2_t._ref_ina)
    var["soma"]['I_K'].record(soma(SOMA_RELATIVE_SEG).k_ion._ref_ik)
    var["soma"]['I_Na'].record(soma(SOMA_RELATIVE_SEG).na_ion._ref_ina)
    var["soma"]['I_Ca'].record(soma(SOMA_RELATIVE_SEG).ca_ion._ref_ica)

    var["soma"]["g_CaHVA"].record(soma(SOMA_RELATIVE_SEG).Ca_HVA._ref_gCa_HVA)
    var["soma"]["g_CaLVA"].record(soma(SOMA_RELATIVE_SEG).Ca_LVAst._ref_gCa_LVAst)
    var["soma"]["g_SK"].record(soma(SOMA_RELATIVE_SEG).SK_E2._ref_gSK_E2)
    var["soma"]["g_SKv3"].record(soma(SOMA_RELATIVE_SEG).SKv3_1._ref_gSKv3_1)
    var["soma"]["g_Im"].record(soma(SOMA_RELATIVE_SEG).Im._ref_gIm)
    var["soma"]["g_Ih"].record(soma(SOMA_RELATIVE_SEG).Ih._ref_gIh)
    var["soma"]["g_NaTs2_t"].record(soma(SOMA_RELATIVE_SEG).NaTs2_t._ref_gNaTs2_t)

    var["soma"]["m_CaHVA"].record(soma(SOMA_RELATIVE_SEG).Ca_HVA._ref_m)
    var["soma"]["h_CaHVA"].record(soma(SOMA_RELATIVE_SEG).Ca_HVA._ref_h)
    var["soma"]["m_CaLVA"].record(soma(SOMA_RELATIVE_SEG).Ca_LVAst._ref_m)
    var["soma"]["h_CaLVA"].record(soma(SOMA_RELATIVE_SEG).Ca_LVAst._ref_h)
    var["soma"]["z_SK"].record(soma(SOMA_RELATIVE_SEG).SK_E2._ref_z)
    var["soma"]["m_SKv3"].record(soma(SOMA_RELATIVE_SEG).SKv3_1._ref_m)
    var["soma"]["m_Im"].record(soma(SOMA_RELATIVE_SEG).Im._ref_m)
    var["soma"]["m_Ih"].record(soma(SOMA_RELATIVE_SEG).Ih._ref_m)
    var["soma"]["m_NaTs2_t"].record(soma(SOMA_RELATIVE_SEG).NaTs2_t._ref_m)
    var["soma"]["h_NaTs2_t"].record(soma(SOMA_RELATIVE_SEG).NaTs2_t._ref_h)

    if stims:
        for key in stims:
            if key == 'syn_gen':
                var['stims']['I_' + key].record(stims[key]._ref_i)

    return var


def run_nrn_simulation():
    h.run()
