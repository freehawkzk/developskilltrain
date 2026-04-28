"""timing_check.py
ACE 时序参数检查。
翻译自 MATLAB: ACE/CommonFunctions/timing_check.m
"""
import copy


def timing_check(org_map: dict) -> dict:
    """检查并设置刺激率、脉宽及每帧脉冲数。

    MATLAB 原始代码中简化路径（未调用 check_timing_parameters），
    直接将 IPG 赋值给 pulses_per_frame_per_channel。
    """
    mod_map = copy.deepcopy(org_map)

    num_selected = org_map['Left']['Nmaxima']
    pw   = org_map['Left']['PulseWidth']
    rate = org_map['Left']['StimulationRate']
    ipg  = org_map['Left']['IPG']

    # MATLAB 简化路径: rate_outL = rate; pw_outL = pw; ppfpchL = ipg;
    mod_map['Left']['StimulationRate'] = rate
    mod_map['Left']['PulseWidth']      = pw
    mod_map['Left']['pulses_per_frame_per_channel'] = ipg
    mod_map['Left']['pulses_per_frame'] = ipg * num_selected

    return mod_map
