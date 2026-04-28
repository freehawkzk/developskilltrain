"""lgf_utils.py
响度增长函数 (Loudness Growth Function) 相关工具。
翻译自 MATLAB: ACE/CommonFunctions/LGF_Q_diff.m 和 initialize_ACE.m 中的 LGF_alpha。
"""
import numpy as np
from scipy.optimize import brentq


def lgf_q(alpha: float, base_level: float, sat_level: float) -> float:
    """计算 LGF Q 因子（相对 sat_level 下降 10 dB 时的输出百分比减少量）。"""
    input_level = sat_level / np.sqrt(10)  # 低于 sat_level 10 dB
    r = (input_level - base_level) / (sat_level - base_level)
    r = float(np.clip(r, 0.0, 1.0))
    v = np.log(1 + alpha * r) / np.log(1 + alpha)
    return float(100.0 * (1.0 - v))


def lgf_q_diff(log_alpha: float, Q: float, base_level: float, sat_level: float) -> float:
    """LGF_Q_diff：供 brentq 求零点使用。"""
    alpha = np.exp(log_alpha)
    return lgf_q(alpha, base_level, sat_level) - Q


def lgf_alpha(Q: float, base_level: float, sat_level: float) -> float:
    """计算使 LGF Q 因子等于指定值 Q 的 alpha 参数。
    翻译自 MATLAB: initialize_ACE.m 中的 LGF_alpha。
    """
    log_a = 0.0
    while True:
        log_a += 1.0
        if lgf_q_diff(log_a, Q, base_level, sat_level) < 0:
            break
    interval = [log_a - 1.0, log_a]
    log_a_zero = brentq(lgf_q_diff, interval[0], interval[1],
                        args=(Q, base_level, sat_level))
    return float(np.exp(log_a_zero))
