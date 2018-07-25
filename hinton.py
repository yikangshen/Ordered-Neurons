# coding=utf-8
from __future__ import print_function
import numpy as np
chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


class BarHack(str):

    def __str__(self):
        return self.internal

    def __len__(self):
        return 1


def plot(arr, max_val=None):
    if max_val is None:
        max_arr = arr
        max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))

    opts = np.get_printoptions()
    np.set_printoptions(edgeitems=500)
    s = str(np.array2string(arr,
                          formatter={
                              'float_kind': lambda x: visual(x, max_val),
                              'int_kind': lambda x: visual(x, max_val)},
                          max_line_width=5000
                          ))
    np.set_printoptions(**opts)

    return s


def visual(val, max_val):
    if abs(val) == max_val:
        step = len(chars) - 1
    else:
        step = int(abs(float(val) / max_val) * len(chars))
    colourstart = ""
    colourend = ""
    if val < 0:
        colourstart, colourend = '\033[90m', '\033[0m'
    return colourstart + chars[step] + colourend