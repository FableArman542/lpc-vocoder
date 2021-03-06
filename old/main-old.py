import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from transmitterold import run_whole_signal, get_values
from receptorold import synthesize

# Radiacao nos labios e uma derivada
# Contrario do pre enfase

wa = 256
ws = 160
pf = 146
thu = 0.45
p = 10
mu = .95

dirak = np.zeros(512)
dirak[0] = 1

sinal = read("../resources/car_nor.wav")[1]
rate = read("../resources/car_nor.wav")[0]
maximo = max(sinal)
# print("MAXIMO", maximo)
sinal = sinal/2**15

# plt.title("sinal")
# plt.plot(sinal)
# plt.show()


# Teste para uma window
# ni = 160*33
# window = sinal[ni: ni+wa]
# get_values(window, thu, p, pf, mu, to_plot=True)

gain_bits = 5
g_max = 2

t_quantization, ak, auxs = run_whole_signal(sinal, ws, thu, p, pf, wa, mu, gain_bits=gain_bits, g_max=g_max, to_plot=False)
synthesize(t_quantization, gain_bits, wa, ws, rate, ak, auxs)


