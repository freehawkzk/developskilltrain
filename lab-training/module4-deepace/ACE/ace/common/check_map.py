"""check_map.py
检查 ACE MAP 参数。
翻译自 MATLAB: ACE/CommonFunctions/check_map.m
"""
from .timing_check import timing_check
from .level_check  import level_check


def check_map(original_map: dict) -> dict:
    """对 MAP 进行时序与幅度级别检查。"""
    timed  = timing_check(original_map)
    leveled = level_check(timed)
    return leveled
