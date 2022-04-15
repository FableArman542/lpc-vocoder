import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.linalg import toeplitz
from scipy import linalg, signal
import numpy.random as rd
from scipy.io.wavfile import read, write


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


def synthesize(pitch, gain, ak, wa, p, rate, maximo):
    # _signal = np.zeros(int(len(pitch) * wa) + 1).tolist()
    gains = calculate_gains(pitch, gain, wa)

    plt.title("Pitch")
    plt.plot(pitch)
    plt.show()

    # Pulso Glotal
    glottal_pulse = []
    g_pulses = []

    new_signal = np.array([])

    new_wa = wa
    for i in range(len(pitch)):
        # Trama nao voziada
        if pitch[i] == 0:
            blank_noise = rd.normal(0, .01, size=new_wa)
            new_signal = np.append(new_signal, blank_noise)
            new_wa = 256
        else:
            # Trama voziada
            current_pulse = np.array([])
            for n in range(int(pitch[i])):
                n_op = .66 * pitch[i]
                if 0 <= n < n_op:
                    formula = ((2 * n_op - 1) * n - 3 * (n ** 2)) / (n_op ** 2 - 3 * n_op + 2)
                    # formula *= gains[i]
                    glottal_pulse.append(formula)
                    current_pulse = np.append(current_pulse, formula)
                elif n_op <= n:
                    glottal_pulse.append(0)
                    current_pulse = np.append(current_pulse, 0)

            # Adicionar pulsos ate wa
            l = int(pitch[i])
            q = int(np.ceil(256/pitch[i]))
            delta = (q * l) - 256

            print(new_wa, "quantidade * length", q*l, delta)
            new_wa = 256 - delta
            current_pulse = current_pulse.flatten()
            for j in range(q):
                new_signal = np.append(new_signal, current_pulse)
                new_signal = new_signal.flatten()

            g_pulses.append(current_pulse)

    y = np.zeros(len(ak))
    zf = np.zeros_like(ak[0])

    length = np.arange(0, len(glottal_pulse), wa)

    print("-----LENGTH-----")
    print("pulso glotal:", len(glottal_pulse))
    print("ak:", len(ak))
    print("g linha:", len(gains))

    # signal.lfilter()
    # y[i], zf[i] = signal.lfilter(b=[1], a=ak[i], x=pulso_glotal, zi= zf[i])

    plt.title("Ganho")
    plt.plot(gains)
    plt.show()

    a = np.int16(new_signal * maximo)
    write('ficheiro.wav', rate, a)

    plt.title("Sinal")
    plt.plot(new_signal)

    x_min = 1100
    x_max = 1400
    y_min = -1.2
    y_max = .5
    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()
