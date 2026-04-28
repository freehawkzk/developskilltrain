"""logarithmic_compression.py
对数压缩（Loudness Growth Function 处理）。
翻译自 MATLAB: ACE/CommonFunctions/logarithmic_compression.m
"""
import numpy as np


def logarithmic_compression(p: dict, u: np.ndarray):
    """对数压缩函数。

    Parameters
    ----------
    p : dict
        包含以下字段：
        - 'BaseLevel'       : 映射到 0 的输入幅值
        - 'SaturationLevel' : 映射到 1 的输入幅值（饱和）
        - 'lgf_alpha'       : 曲线形状因子
        - 'sub_mag'         : 低于 BaseLevel 时的输出值（负数）
    u : np.ndarray
        输入幅值矩阵（任意形状）。

    Returns
    -------
    v   : np.ndarray  输出（范围 sub_mag .. 1）
    sub : np.ndarray  bool，标记低于 BaseLevel 的位置
    sat : np.ndarray  bool，标记高于 SaturationLevel 的位置
    """
    u = np.array(u, dtype=float)
    base = p['BaseLevel']
    sat_l = p['SaturationLevel']
    alpha = p['lgf_alpha']
    sub_mag = p['sub_mag']

    r = (u - base) / (sat_l - base)

    sat = r > 1.0
    r = np.where(sat, 1.0, r)

    sub = r < 0.0
    r = np.where(sub, 0.0, r)

    v = np.log(1.0 + alpha * r) / np.log(1.0 + alpha)
    v = np.where(sub, sub_mag, v)

    return v, sub, sat
