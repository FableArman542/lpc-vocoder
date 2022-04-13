import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.linalg import toeplitz
from scipy import linalg, signal
import numpy.random as rd


def calculate_gains(pitch, gain, wa):
    gains = np.zeros_like(gain)

    for i in range(len(pitch)):
        # Trama nao voziada
        if pitch[i] == 0:
            gains[i] = gain[i] * np.sqrt(1 / wa)
        else:
            # Trama voziada
            gains[i] = gain[i] * np.sqrt(pitch[i] / wa)

    return gains


def synthesize(pitch, gain, ak, wa, p):
    _signal = np.zeros(int(len(pitch) * wa) + 1).tolist()
    gains = calculate_gains(pitch, gain, wa)

    # Pulso Glotal
    glottal_pulse = []

    signal_index = 0
    for i in range(len(pitch)):
        # Trama nao voziada
        if pitch[i] == 0:
            blank_noise = rd.normal(loc=0, scale=.5, size=wa)
            _signal[signal_index] = blank_noise
        else:
            # Trama voziada
            for n in range(int(pitch[i])):
                n_op = .66 * pitch[i]
                if 0 <= n < n_op:
                    formula = ((2 * n_op - 1) * n - 3 * (n ** 2)) / (n_op ** 2 - 3 * n_op + 2)
                    glottal_pulse.append(formula)
                elif n_op <= n:
                    glottal_pulse.append(0)

        signal_index += wa

    y = np.zeros(len(ak))
    zf = np.zeros_like(ak[0])

    length = np.arange(0, len(glottal_pulse), wa)

    print("-----LENGTH-----")
    print("pulso glotal:", len(glottal_pulse))
    print("ak:", len(ak))
    print("g linha:", len(gains))

    # signal.lfilter()
    # y[i], zf[i] = signal.lfilter(b=[1], a=ak[i], x=pulso_glotal, zi= zf[i])

    # for i in length:
    #     if i is not length[-1]:
    #         y[i]

    # x =  trama filtro glotal
    # y = np.zeros(len(ak))
    # zf = np.zeros_like(ak[0])
    # for i in range(len(ak)):
    #     y[i], zf[i] = signal.lfilter(gains[i], ak[i], )
    #
    #
    #     signal_index += wa

    # s = []
    # idx = 0
    # for i in range(len(pitch)):
    #     conta = []
    #     for k in range(1, p):
    #         ak * s[n-k]
    #     idx += wa

    plt.title("Ganho")
    plt.plot(gains)
    plt.show()

    plt.title("Pulso Glotal")
    plt.plot(glottal_pulse)
    x_min = 0
    x_max = 200
    y_min = -1.2
    y_max = .5
    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()
