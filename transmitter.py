import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.linalg import toeplitz
from scipy import linalg, signal
import numpy.random as rd
import struct
import re
from utils.utils import int2bin, write_to_file, read_file

dirak = np.zeros(512)
dirak[0] = 1


def hamming_window(w):
    j = w * np.hamming(len(w))
    j -= np.mean(w)
    return j


def autocorrelation(s, pf):
    r = np.zeros(pf)

    for k in range(pf):
        rk = 0
        for n in range(k, pf):
            rk += s[n] * s[n - k]
        r[k] = rk

    energia = r[0]

    return r / r[0], energia, r


'''
    Calcular parametros para uma trama
'''


def get_values(window, thu, p, pf, mu, to_plot=False):
    r, e, non_normalized_r = autocorrelation(window, pf)

    if to_plot:
        plt.title('Autocorrelacao da trama')
        plt.plot(r)
        plt.grid(True)
        plt.show()

    vibration = pitch(r, e, 5, thu)
    if to_plot:
        print("Voziamento:", vibration)

    window = hamming_window(window)

    # Adicionar o pre enfase
    if vibration != 0:
        window = window[2:] - mu * window[1:-1]
        r, e, non_normalized_r = autocorrelation(window, pf)

    if to_plot:
        plt.title("Trama com janela de hamming")
        plt.plot(window)
        plt.show()

    m = toeplitz(r[:p])
    mr = np.array(r[1:p + 1]) * -1
    minv = linalg.inv(m)
    a = np.dot(mr, minv)

    _gain = gain(non_normalized_r, a, p)
    if to_plot:
        print("Ganho:", _gain)

    if to_plot:
        resposta_impulsiva = signal.lfilter([_gain], np.hstack(([1], a)), dirak)
        r_ri, e_ri, n_r_ri = autocorrelation(resposta_impulsiva, pf)

        plt.title("Autocorrelação da resposta impulsiva e da janela")
        plt.plot(r_ri)
        plt.plot(r)
        plt.grid(True)
        plt.show()

    if to_plot:
        _r = 20 * np.log10(np.abs(np.fft.fft(r, 512)))
        _n_r = 20 * np.log10(np.abs(np.fft.fft(r_ri, 512)))

        plt.title("Espetro")
        plt.plot(_r[:int(len(_r) / 2)])
        plt.plot(_n_r[:int(len(_n_r) / 2)])
        plt.show()

    return r, non_normalized_r, vibration, e, _gain, a


'''
    Calcular parametros para o sinal inteiro
'''


def run_whole_signal(_signal, ws, thu, p, pf, wa, mu, gain_bits, g_max, to_plot=False):
    vibrations = np.zeros(int(len(_signal) / ws))
    energy = np.zeros(int(len(_signal) / ws))
    gain = np.zeros(int(len(_signal) / ws))
    ak = np.zeros((int(len(_signal) / ws), 10))
    # ak = ak.tolist()

    for i in range(0, int(len(_signal) / ws)):
        ni = i * ws
        window = _signal[ni:ni + wa]
        _, _, v, e, g, a = get_values(window, thu, p, pf, mu)

        vibrations[i] = v
        energy[i] = e
        gain[i] = g
        ak[i] = a

    correct_pitch(vibrations)

    t_quantization = quantize(vibrations, gain, gain_bits, g_max)

    if to_plot:
        print(np.argmax(energy))
        plt.title("Energia")
        plt.plot(energy)
        plt.show()

        plt.title("Voziamento")
        plt.plot(vibrations)
        plt.show()

        plt.title("Ganho")
        plt.plot(gain)
        plt.show()

    return t_quantization, ak


def pitch(r, e, t, threshold, error=0.01, debug=False):
    if e < error: return 0

    val = np.argmax(r[20::]) + 20 if np.max(r[20::]) > threshold else 0

    # Remover o duplo pitch

    if debug: print("Valor:", val)
    val_min = int((np.argmax(r[20::]) + 20) / 2 - t)
    val_max = int((np.argmax(r[20::]) + 20) / 2 + t)

    if debug: print("Valor Minimo:", val_min, "| Valor Maximo:", val_max)

    cut_sinal = np.zeros_like(r)
    cut_sinal[val_min:val_max] = r[val_min:val_max]

    new_val = np.argmax(cut_sinal)
    n_val = np.max(cut_sinal)

    if new_val > 20 and n_val > threshold:
        if debug:
            print("Novos valores: ", new_val, n_val)
        val = new_val

    return val


def gain(r, a, p):
    G = r[0]
    for k in range(1, p + 1):
        G += a[k - 1] * r[k]
    return np.sqrt(G)


def correct_pitch(vibrations):
    for i in range(len(vibrations)):
        if i + 2 <= len(vibrations) - 1:
            v1 = vibrations[i]
            v2 = vibrations[i + 1]
            v3 = vibrations[i + 2]
            if v1 == v3 and v2 != v1 and v1 != 0.:
                vibrations[i + 1] = (v1 + v3) / 2
            elif v1 == v3 and v2 != v1 and v1 == 0.:
                vibrations[i + 1] = 0.
        else:
            break


def quantize(pitch, gains, gain_bits, g_max):
    quantized_pitch = quantize_pitch(pitch)

    t_quantization, indexes = quantize_gain(gains, g_max, gain_bits, plot=False)
    gain_quantization = int2bin(indexes, gain_bits)

    write_to_file(filename="pitches", bits=str(quantized_pitch))
    write_to_file(filename="gains", bits=''.join(str(g) for g in gain_quantization))

    return t_quantization


def quantize_pitch(pitch):
    new_pitch = ""
    for i in range(len(pitch)):
        new_pitch += "{0:07b}".format(int(pitch[i] - 19)) if pitch[i] != 0 else "{0:07b}".format(int(0))
    return new_pitch


def verify(d, Vd, Vq, lvl):
    I = d <= np.array(Vd)
    if d > Vd[-1]:
        S = Vq[-1]
        index = lvl[-1]
    else:
        S = Vq[I][0]
        index = lvl[I][0]
    return S, index


def quantiz(tquantization, tdecision, x, lvl):  # Iq,Vq,x
    func = np.vectorize(verify, excluded=['Vd', 'Vq', 'lvl'])
    res, indexes = func(x, Vd=tdecision, Vq=tquantization, lvl=lvl)
    return res, indexes


def quantize_gain(gains, g_max, bits, plot=False):
    delta = g_max / 2 ** bits
    # gains = np.arange(0, 1, .01)
    lvl = np.arange(0, 2 ** bits)

    tquantization = np.arange(delta / 2, g_max, delta)  # Decisao
    tdecision = np.arange(0 + delta, g_max + delta, delta)  # Quantificao

    quantized, indexes = quantiz(tquantization, tdecision, gains, lvl)

    if plot:
        plt.plot(gains)
        plt.plot(tquantization[indexes])
        plt.grid(True)
        plt.show()

    return tquantization, indexes
