# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:22:31 2019

@author: Leonardo Melo
"""

import numpy as np

import scipy.signal as sg

def lpc_to_lsf(all_lpc):
    if len(all_lpc.shape) < 2:
        all_lpc = all_lpc[None]
    order = all_lpc.shape[1] - 1
    all_lsf = np.zeros((len(all_lpc), order))
    for i in range(len(all_lpc)):
        lpc = all_lpc[i]
        lpc1 = np.append(lpc, 0)
        lpc2 = lpc1[::-1]
        sum_filt = lpc1 + lpc2
        diff_filt = lpc1 - lpc2

        if order % 2 != 0:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, -1])
            deconv_sum, _ = sg.deconvolve(sum_filt, [1, 1])

        roots_diff = np.roots(deconv_diff)
        roots_sum = np.roots(deconv_sum)
        angle_diff = np.angle(roots_diff[::2])
        angle_sum = np.angle(roots_sum[::2])
        lsf = np.sort(np.hstack((angle_diff, angle_sum)))
        if len(lsf) != 0:
            all_lsf[i] = lsf
    return np.squeeze(all_lsf)


def lsf_to_lpc(all_lsf):
    if len(all_lsf.shape) < 2:
        all_lsf = all_lsf[None]
    order = all_lsf.shape[1]
    all_lpc = np.zeros((len(all_lsf), order + 1))
    for i in range(len(all_lsf)):
        lsf = all_lsf[i]
        zeros = np.exp(1j * lsf)
        sum_zeros = zeros[::2]
        diff_zeros = zeros[1::2]
        sum_zeros = np.hstack((sum_zeros, np.conj(sum_zeros)))
        diff_zeros = np.hstack((diff_zeros, np.conj(diff_zeros)))
        sum_filt = np.poly(sum_zeros)
        diff_filt = np.poly(diff_zeros)

        if order % 2 != 0:
            deconv_diff = sg.convolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff = sg.convolve(diff_filt, [1, -1])
            deconv_sum = sg.convolve(sum_filt, [1, 1])

        lpc = .5 * (deconv_sum + deconv_diff)
        # Last coefficient is 0 and not returned
        all_lpc[i] = lpc[:-1]
    return np.squeeze(all_lpc)