import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.linalg import toeplitz
from scipy import linalg, signal
import numpy.random as rd

dirak = np.zeros(512)
dirak[0] = 1


def hamming_window(w):
    j = w * np.hamming(len(w))
    j -= np.mean(w)
    return j


def autocorrelation(sinal, wa, ni, pf):
    r = np.zeros(pf)

    for k in range(pf):
        rk = 0
        for n in range(k, pf):
            rk += sinal[n] * sinal[n - k]
        r[k] = rk

    energia = r[0]

    return r / r[0], energia, r


def get_values(window, thu, p, pf, wa, ni, mu, to_plot=False):
    r, e, non_normalized_r = autocorrelation(window, wa, ni, pf)

    if to_plot:
        plt.title('Autocorrelacao da trama em ni = ' + str(ni))
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

    if to_plot:
        plt.title("Trama com janela de hamming")
        plt.plot(window)
        plt.show()

    m = toeplitz(r[:p])
    mr = np.array(r[1:p + 1]) * -1
    minv = linalg.inv(m)
    a = np.dot(mr, minv)

    gain = ganho(non_normalized_r, a, p)
    if to_plot:
        print("Ganho:", gain)

    if to_plot:
        resposta_impulsiva = signal.lfilter([gain], np.hstack(([1], a)), dirak)
        r_ri, e_ri, n_r_ri = autocorrelation(resposta_impulsiva, wa, 0, pf)

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

    return r, non_normalized_r, vibration, e, gain, a


def run_whole_signal(signal, ws, thu, p, pf, wa, mu, to_plot=False):
    vibrations = np.zeros(int(len(signal) / ws) + 1)
    energy = np.zeros(int(len(signal) / ws) + 1)
    gain = np.zeros(int(len(signal) / ws) + 1)
    ak = np.zeros(int(len(signal) / ws) + 1)
    ak = ak.tolist()

    for i in range(0, int((len(signal) - 256) / ws)):
        ni = i * ws
        window = signal[ni:ni + wa]
        _, _, v, e, g, a = get_values(window, thu, p, pf, wa, ni, mu)

        vibrations[i] = v
        energy[i] = e
        gain[i] = g
        # print(type(a), a.shape, a[0])
        ak[i] = a

    correct_pitch(vibrations)

    if to_plot:
        plt.title("Energia")
        plt.plot(energy)
        plt.show()

        plt.title("Voziamento Corrigido")
        plt.plot(vibrations)
        plt.show()

        plt.title("Ganho")
        plt.plot(gain)
        plt.show()

    return vibrations, gain, ak


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
        if debug: print("Novos valores: ", new_val, n_val)
        val = new_val

    return val


def ganho(r, a, p):
    G = r[0]
    for k in range(1, p + 1):
        G += a[k - 1] * r[k]
    return G


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
