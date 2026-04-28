"""map_ace.py
构建 ACE MAP 参数字典。
翻译自 MATLAB: ACE/Map_ACE.m
"""
import numpy as np


def map_ace(voc_paras: dict) -> dict:
    """构建默认 ACE MAP（仅左耳）。

    Parameters
    ----------
    voc_paras : dict
        需包含: NumberOfBands, Nmaxima, THR, MCL, BandGain, StimulationRate

    Returns
    -------
    MAP : dict
        含 'Left' 子字典的 MAP 结构体（与 MATLAB Map_ACE 等价）。
    """
    n = int(voc_paras['NumberOfBands'])
    MAP = {
        'Left': {
            'ImplantType':       'CI24RE',
            'SamplingFrequency': 16000,
            'Strategy':          'ACE',
            'Nmaxima':           int(voc_paras['Nmaxima']),
            'StimulationMode':   'MP1+2',
            'StimulationRate':   float(voc_paras['StimulationRate']),
            'PulseWidth':        25,
            'IPG':               8,
            'Sensitivity':       2.3,
            'Gain':              25.0,
            'Volume':            10.0,
            'Q':                 20.0,
            'BaseLevel':         0.0156,
            'SaturationLevel':   0.5859,
            'ChannelOrderType':  'base-to-apex',
            'FrequencyTable':    'Default',
            'Window':            'Hanning',
            'NumberOfBands':     n,
            # Electrodes: [n, n-1, ..., 1]（电极号，最低号=最高频）
            'Electrodes':        np.arange(n, 0, -1, dtype=float),
            'THR':               float(voc_paras['THR'])  * np.ones(n),
            'MCL':               float(voc_paras['MCL'])  * np.ones(n),
            'BandGains':         float(voc_paras['BandGain']) * np.ones(n),
        }
    }
    return MAP
