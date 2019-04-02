# This function synthesizes two time domain signals containing arbitrary frequency information at frequencies
# [-31, -29, ... -1, 1, ... 29, 31]. This is intended to model the time-domain initial state sampled and
# filtered in the quantum emulation device.

import numpy as np
import matplotlib.pyplot as plt

def SynthesizeSignals(fs, QED_frequencies):

    # Interleave zeros
    fmin = 1000               # Lowest frequency of the system. We want to generate one period of the lowest frequency
    N = int(fs/fmin)          # N = fs/fmin gives number of samples in one period. Should be a power of two for radix-2 FFT's
    x_k = np.zeros(N,dtype=complex) # x_k will be the frequency-domain vector representing a QED state i.e. x[k]. By our convention x[0] corresponds
                              # to the frequency bin for -fs/2  and x[N] corresponds to the fs/2 bin.

    for i in range(0,len(x_k)):
        Res = fs/N            # FFT resolution. Also a step size
        f_i = -fs/2 + Res*i
        if np.isin(f_i,QED_frequencies):                  # if f_i is a QED frequency
            x_k[i] = N*(np.random.randn() + 1j*np.random.randn()) # Frequency information is random complex number for now. Just simulated data
                                                                  # Applied a scaling of N for DFT normalization
    x_k = np.fft.ifftshift(x_k) # Convert data to 'standard' form for numpy.fft function (see numpy documentation)
    x_n = np.fft.ifft(x_k)
    x_k = (np.fft.fftshift(x_k/N))  # Convert back for plotting convenience

    # Plotting parameters
    plt.figure(1)
    t = np.arange(0,N,1);
    t = (1/fs)*t
    plt.plot(t, np.real(x_n),'r',label='Real')
    plt.plot(t, np.imag(x_n),'b',label='Imag')
    plt.legend()
    plt.title('Time Domain Sequence')

    plt.figure(2)
    nyquist = int(fs/2)
    res = 1000
    f = np.arange(-nyquist, nyquist, res)
    plt.stem(f, np.real(x_k),'r',label='Real')
    plt.stem(f, np.imag(x_k),'b',label='Imag')
    plt.title('Frequency Domain Sequence')
    axis_elem = len(QED_frequencies) - 1
    xmin = QED_frequencies[0] - 1000
    xmax = QED_frequencies[axis_elem] + 1000
    plt.axis([xmin, xmax, -3, 3])
    plt.legend()
    plt.show()

    return x_n, x_k

