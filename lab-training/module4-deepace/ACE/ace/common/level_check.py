"""level_check.py
ACE 阈值/舒适度级别检查。
翻译自 MATLAB: ACE/CommonFunctions/level_check.m
"""
import copy
import numpy as np


def level_check(org_map: dict) -> dict:
    """检查并修正 THR / MCL 值。"""
    mod_map = copy.deepcopy(org_map)

    for side in ('Left', 'Right'):
        if side not in org_map:
            continue
        n   = org_map[side]['NumberOfBands']
        thr = list(mod_map[side]['THR'].astype(float))
        mcl = list(mod_map[side]['MCL'].astype(float))

        for i in range(n):
            if thr[i] < 0:
                print(f'Channel {i+1}: THR = {thr[i]} -> corrected to 0.')
                thr[i] = 0.0
            if mcl[i] > 255:
                print(f'Channel {i+1}: MCL = {mcl[i]} -> corrected to 255.')
                mcl[i] = 255.0
            if thr[i] > mcl[i]:
                print(f'Channel {i+1}: THR({thr[i]}) > MCL({mcl[i]}) -> THR corrected to 0.')
                thr[i] = 0.0

        mod_map[side]['THR'] = np.array(thr)
        mod_map[side]['MCL'] = np.array(mcl)

    return mod_map
