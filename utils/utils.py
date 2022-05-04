import numpy as np
from array import array
import re


def int2bin(N, nBits):
    return np.hstack(list(map(lambda x: list(np.binary_repr(x, nBits)), N))).astype('int')


def bin2int(N, nBits):
    intArr = []
    for i in range(int(len(N) / nBits)):
        newInt = 0
        for j in range(nBits):
            newInt += (int(N[j + (i * nBits)]) * 2 ** (nBits - j - 1))
        intArr.append(newInt)
    return intArr


def find_padding(a):
    d = a % 8
    return 8 - d if d != 0 else 0


def write_to_file(filename, bits):
    bin_array = array("B")
    d = find_padding(len(bits))

    for octect in re.findall(r'\d{1,8}', bits):
        if len(octect) < 8:
            ajuda = ''.join(str(e) for e in np.zeros(d).astype(int))
            novo = ajuda + octect[::-1]
            bin_array.append(int(novo, 2))
        else:
            bin_array.append(int(octect[::-1], 2))

    with open(filename + ".BIN", "wb") as f:
        f.write(bytes(bin_array))
        f.close()


def read_file(filename, r):
    dtype = np.dtype('B')
    try:
        with open(filename + ".BIN", "rb") as f:
            numpy_data = np.fromfile(f, dtype)
        bits_in_array = np.unpackbits(np.frombuffer(numpy_data, np.dtype('B')))
        ajuda = ''
        for b in range(1, int(len(bits_in_array) / 8 + 1)):
            ajuda += ''.join(str(e) for e in bits_in_array[8 * (b - 1):8 * b][::-1])

        converted = np.array(bin2int(ajuda, r)).astype(int)
        return converted
    except IOError:
        print('Error While Opening the file!')
