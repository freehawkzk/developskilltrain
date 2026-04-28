"""load_map.py
加载并验证 ACE MAP。
翻译自 MATLAB: ACE/CommonFunctions/load_map.m
"""
from ..map_ace   import map_ace
from .check_map  import check_map


def load_map(voc_paras: dict) -> dict:
    """构建并验证 ACE MAP。

    Parameters
    ----------
    voc_paras : dict
        声码器参数: NumberOfBands, Nmaxima, THR, MCL, BandGain, StimulationRate

    Returns
    -------
    p : dict  经过检查的 MAP。
    """
    MAP = map_ace(voc_paras)
    p   = check_map(MAP)
    return p
