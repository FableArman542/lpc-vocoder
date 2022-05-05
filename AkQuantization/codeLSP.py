# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:44:33 2019

@author: Rodrigo Moura
"""
import numpy as np
vd1=[0.0169,0.0247,    0.0297,    0.0331,    0.0387,    0.0475 ,   0.0575]
vd2=[0.0278 ,   0.0313 ,   0.0350,    0.0387 ,   0.0428  ,  0.0475  ,  0.0525  ,  0.0575  ,  0.0625 ,0.0675 ,   0.0731 ,   0.0800 ,   0.0881  ,  0.0969 ,   0.1056]
vd3=[0.0550 ,   0.0600 ,   0.0650 ,   0.0703  ,  0.0766 ,   0.0841,    0.0925  ,  0.1016 ,   0.1125, 0.1250  ,  0.1375 ,   0.1500 ,   0.1625 ,   0.1750  ,  0.1875]
vd4=[0.0800 ,   0.0863 ,   0.0947,    0.1047 ,   0.1156 ,   0.1281 ,   0.1406  ,  0.1525 ,   0.1650, 0.1775  ,  0.1900 ,   0.2025 ,   0.2150 ,   0.2275 ,   0.2400]
vd5=[0.1281  ,  0.1363 ,   0.1463 ,   0.1559  ,  0.1647 ,   0.1737 ,   0.1837  ,  0.1937  ,  0.2037, 0.2138  ,  0.2250 ,   0.2375 ,   0.2500 ,   0.2625  ,  0.2750]
vd6=[0.1900 ,   0.2037  ,  0.2200  ,  0.2394  ,  0.2625 ,   0.2875 ,   0.3125]
vd7=[0.2300 ,   0.2400  ,  0.2537 ,   0.2750  ,  0.2987 ,   0.3237 ,   0.3500]
vd8=[0.2891  ,  0.3078 ,   0.3234  ,  0.3406  ,  0.3594 ,   0.3813 ,   0.4062]
vd9=[0.3525  ,  0.3675  ,  0.3812  ,  0.3937  ,  0.4069 ,   0.4213 ,   0.4362]
vd10=[0.4037 ,   0.4138 ,   0.4231  ,  0.4319 ,   0.4425 ,   0.4562 ,   0.4712]

vq1=[0.0125 ,   0.0213 ,   0.0281 ,   0.0313 ,   0.0350 ,   0.0425,    0.0525  ,  0.0625]
vq2=[0.0263 ,   0.0294 ,   0.0331 ,   0.0369 ,   0.0406  ,  0.0450 ,   0.0500 ,   0.0550,    0.0600, 0.0650  ,  0.0700  ,  0.0763  ,  0.0838   , 0.0925 ,   0.1013 ,   0.1100]
vq3=[0.0525  ,  0.0575 ,   0.0625 ,   0.0675 ,   0.0731 ,   0.0800 ,   0.0881  ,  0.0969,    0.1062 ,0.1187 ,   0.1312  ,  0.1438 ,   0.1563,    0.1687  ,  0.1812  ,  0.1937]
vq4=[0.0775  ,  0.0825  ,  0.0900 ,   0.0994 ,   0.1100 ,   0.1212,    0.1350  ,  0.1463 ,   0.1588,    0.1712  ,  0.1837  ,  0.1962 ,   0.2088 ,   0.2213  ,  0.2338  ,  0.246]
vq5=[0.1250 ,   0.1312 ,  0.1413  ,  0.1513  ,  0.1606,    0.1687 ,   0.1787  ,  0.1887, 0.1987,    0.2088 ,   0.2188  ,  0.2313  ,  0.2437 ,   0.2562  ,  0.2687  ,  0.2813]
vq6=[0.1837 ,   0.1962 ,   0.2113 ,   0.2288 ,   0.2500 ,   0.2750  ,  0.3000  ,  0.3250]
vq7=[0.2250 ,   0.2350  ,  0.2450 ,   0.2625 ,   0.2875 ,   0.3100 ,   0.3375 ,   0.3625]
vq8=[0.2781  ,  0.3000  ,  0.3156 ,   0.3312  ,  0.3500 ,   0.3688  ,  0.3937  ,  0.4188]
vq9=[0.3450  ,  0.3600  ,  0.3750  ,  0.3875  ,  0.4000 ,   0.4138  ,  0.4288  ,  0.4437]
vq10=[0.3987  ,  0.4088 ,   0.4188  ,  0.4275  ,  0.4363  ,  0.4487  ,  0.4637  ,  0.4788]


def quantize(signal, partitions, codebook):
    indices = []
    quanta = []
    signal=[signal]
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        indices.append(index)
        quanta.append(codebook[index])
    return indices[0], quanta[0]


def codeLSP(lsp):
    """
    Recebe uma trama de lsp  

    Returns
    =======
    Array de strings;
    
    """
    bitsLSP=[]
    lspq = np.arange(len(lsp)).astype(np.float)
    #print(lspq)
    index,lspq[0] = quantize(lsp[0], vd1,vq1)
    binary=format(index,'03b')
    bitsLSP.append(binary)
    
    index,lspq[1]=quantize(lsp[1], vd2, vq2)
    while(lspq[1]<lspq[0]):
        index=index+1
        lspq[1]=vq2[index]
    binary=format(index,'04b')    
    bitsLSP.append(binary)
    
    index,lspq[2]=quantize(lsp[2], vd3, vq3)
    while(lspq[2]<lspq[1]):
        index=index+1
        lspq[2]=vq3[index]
    binary=format(index,'04b')    
    bitsLSP.append(binary)
    
    
    index,lspq[3]=quantize(lsp[3], vd4, vq4)
    while(lspq[3]<lspq[2]):
        index=index+1
        lspq[3]=vq4[index]
    binary=format(index,'04b')    
    bitsLSP.append(binary)
    
    
    index,lspq[4]=quantize(lsp[4], vd5, vq5)
    while(lspq[4]<lspq[3]):
        index=index+1
        lspq[4]=vq5[index]
    binary=format(index,'04b')    
    bitsLSP.append(binary)
    
    index,lspq[5]=quantize(lsp[5], vd6, vq6)
    while(lspq[5]<lspq[4]):
        index=index+1
        lspq[5]=vq6[index]
    binary=format(index,'03b')    
    bitsLSP.append(binary)
    
    
    index,lspq[6]=quantize(lsp[6], vd7, vq7)
    while(lspq[6]<lspq[5]):
        index=index+1;
        lspq[6]=vq7[index]
    binary=format(index,'03b')    
    bitsLSP.append(binary)
    
    
    index,lspq[7]=quantize(lsp[7], vd8, vq8)
    while(lspq[7]<lspq[6]):
        index=index+1
        lspq[7]=vq8[index]
    binary=format(index,'03b')    
    bitsLSP.append(binary)
    
    
    index,lspq[8]=quantize(lsp[8], vd9, vq9)
    while(lspq[8]<lspq[7]):
        index=index+1
        lspq[8]=vq9[index]
    binary=format(index,'03b')    
    bitsLSP.append(binary)
    
    index,lspq[9]=quantize(lsp[9], vd10, vq10)
    while(lspq[9]<lspq[8]):
        index=index+1
        lspq[9]=vq10[index]
    binary=format(index,'03b')    
    bitsLSP.append(binary)
    
    return bitsLSP, lspq
    
    
    
    
    
    
    
    