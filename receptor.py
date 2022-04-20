import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.linalg import toeplitz
from scipy import linalg, signal
import numpy.random as rd
from scipy.io.wavfile import read, write


def calculate_gains(pitch, gain, wa, plot=False):
    gains = np.zeros_like(gain)

    for i in range(len(pitch)):
        # Trama nao voziada
        if pitch[i] == 0:
            gains[i] = gain[i] * np.sqrt(1 / wa)
        else:
            # Trama voziada
            gains[i] = gain[i] * np.sqrt(pitch[i] / wa)

    if plot:
        plt.plot(gains)
        plt.show()

    return gains


def synthesize(pitch, gain, ak, wa, ws, rate, maximo):

    gains = calculate_gains(pitch, gain, wa)

    # Pulso Glotal
    g_pulses = []

    new_signal = np.array([])

    y = np.array([])
    zf = np.zeros_like(ak[0])

    gl_last = (2/3) * gains[0]

    new_wa = ws
    for i in range(len(pitch)):
        # Trama nao voziada
        if pitch[i] == 0:
            blank_noise = rd.normal(0, .01, size=new_wa)
            blank_noise = blank_noise * gains[i]

            yy, zf = signal.lfilter(b=[1.], a=np.concatenate(([1.], ak[i])), x=blank_noise, zi=zf)  # Filtro
            y = np.append(y, yy)
            new_signal = np.append(new_signal, blank_noise)
            new_wa = ws
        else:
            # Trama voziada
            current_pulse = np.array([])
            n_op = np.ceil(.66 * pitch[i])
            for n in range(int(pitch[i])):
                if 0 <= n < n_op:
                    formula = ((2 * n_op - 1) * n - 3 * (n ** 2)) / (n_op ** 2 - 3 * n_op + 2)
                    # formula = formula * gains[i]
                    current_pulse = np.append(current_pulse, formula)
                elif n_op <= n:
                    current_pulse = np.append(current_pulse, 0)

            # Adicionar pulsos ate ws
            l = int(pitch[i])
            q = int(np.ceil(new_wa / pitch[i]))
            delta = (q * l) - new_wa
            # print(q*l)

            new_wa = ws - delta
            current_pulse = current_pulse.flatten()
            ajuda = np.asarray([])

            for j in range(q):
                # new_signal = np.append(new_signal, current_pulse)
                # new_signal = new_signal.flatten()
                new_g = gains[i] * (j+1)
                new_g += gl_last*(q - (j+1))
                new_g = new_g/(j+1)

                actual_pulse = current_pulse * new_g
                ajuda = np.append(ajuda, actual_pulse)
                ajuda = ajuda.flatten()




            yy, zf = signal.lfilter(b=[1.], a=np.concatenate(([1.], ak[i])), x=ajuda, zi=zf)
            y = np.append(y, yy)

            g_pulses.append(current_pulse)

    y = y.flatten()

    # audio = np.int16(new_signal * maximo)
    # write('ficheiro.wav', rate, audio)

    audio = np.int16(y * maximo)
    write('Output.wav', rate, audio)

def decode():
    return
