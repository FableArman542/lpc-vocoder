import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.linalg import toeplitz
from scipy import linalg, signal
from transmitter import run_whole_signal, get_values
from receptor import synthesize

# Radiacao nos labios e uma derivada
# Contrario do pre enfase

wa = 256
ws = 160
pf = 146
thu = 0.45
p = 10
mu = .95

ni = 1500

dirak = np.zeros(512)
dirak[0] = 1

sinal = read("./resources/car_nor.wav")[1]
rate = read("./resources/car_nor.wav")[0]
maximo = max(sinal)
sinal = sinal/max(sinal)


# Teste para uma window
# window = sinal[ni: ni+wa]
# get_values(window, thu, p, pf, wa, ni, mu)

vibrations, gain, ak = run_whole_signal(sinal, ws, thu, p, pf, wa, mu, to_plot=False)
synthesize(vibrations, gain, ak, wa, p, rate, maximo)



