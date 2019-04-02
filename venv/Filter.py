# The functions in this file perform the modified FFT style filtering

import numpy as np
import matplotlib.pyplot as plt


# Given the time domain function representing an unfiltered projection onto a qubit, this function uses information
# about the target qubit in the projection and whether the signal is a projection onto the zero or the one state of
# the filter to perform the appropriate filtering operation


# Though a recursive function is possible to save operations,
# we resort to demonstrating each operation separately for clarity
# Here, i is the target qubit and x_n is the input sequence.
def DIF_FFT(i,x_n,fs,QED_frequencies):
    N = len(x_n)

    if i == 0: # This is the "least significant" qubit, and requires filtering every other frequency with spacings of 2kHz
        # Example: Stopband: [-8-4,0,4,8]
        #          Passband: [-6,-2,2,6]
        # In effect, we'd like to pass X[4k+2] and stop X[4k+1] and X[4k+3], and X[4k]
        # The savings comes from only computing  X[4k+2] (not X[4k+3], X[4k+1] or X[4k])

        # Make intermediate time-domain sequences
        y1 = np.zeros(int(N/2), dtype=np.complex)

        for i in range(0,int(N/2)):
            y1[i] = x_n[i] + x_n[i + int(N/2)]

        y2 = np.zeros(int(N/4), dtype=np.complex)
        for i in range(0, int(N/4)):
            y2[i] = (y1[i] - y1[i + int(N/4)])*np.exp(-1j*(2*np.pi/(N/2))*i)
        y_k = np.fft.fft(y2)
        y_k = np.fft.fftshift(y_k)

        X_k = np.zeros(N, dtype=np.complex)
        for i in range(0,int(N/4)):
            X_k[4*i] = 0
            X_k[4*i + 1] = 0
            X_k[4*i + 2] = y_k[i]
            X_k[4*i + 3] = 0


        X_k_shift = np.fft.fftshift(X_k)
        x_n = np.fft.ifft(X_k_shift)
        plt.figure(3)
        t = np.arange(0, N, 1);
        plt.plot(t, np.real(x_n), 'r', label='Real')
        plt.plot(t, np.imag(x_n), 'b', label='Imag')
        plt.legend()
        plt.title('Filter Time Domain Sequence')

        plt.figure(4)
        nyquist = int(fs / 2)
        res = 1000
        X_k = X_k/N
        f = np.arange(-nyquist, nyquist, res)
        plt.stem(f, np.real(X_k), 'r', label='Real')
        plt.stem(f, np.imag(X_k), 'b', label='Imag')
        plt.title('Frequency Domain Sequence Filtered')
        axis_elem = len(QED_frequencies) - 1
        xmin = 2 * QED_frequencies[0] - 1000
        xmax = 2 * QED_frequencies[axis_elem] + 1000
        plt.axis([xmin, xmax, -5, 5])
        plt.legend()
        plt.show()




    elif i == 1:
        # In this case, our Passband is [...-9,-7,-1,1,7,9,...]
        # And the stopband is everything else
        # To get the above stopband, we compute
        # X[8k+1] and X[8k+7] (1/4 of the total spectrum)


        # Make intermediate time-domain sequences
        y1 = np.zeros(int(N/2), dtype=np.complex)
        for i in range(0,int(N/2)):
            y1[i] = (x_n[i] - x_n[i + int(N/2)])*np.exp(-1j*2*np.pi/N*i) # Used for X[2k+1]
        y2 = np.zeros(int(N/4), dtype=np.complex)
        for i in range(0, int(N/4)):
            y2[i] = (y1[i] - y1[i + int(N/4)])*np.exp(-1j*(2*np.pi/(N/2))*i) # Used for X[4k+3]
        y3 = np.zeros(int(N/8), dtype=np.complex)
        for i in range(0, int(N/8)):
            y3[i] = (y2[i] - y2[i + int(N/8)])*np.exp(-1j*(2*np.pi/(N/4))*i) # For X[8k + 7]


        y4 = np.zeros(int(N/2), dtype=np.complex)
        for i in range(0,int(N/2)):
            y4[i] = (x_n[i] - x_n[i + int(N/2)])*np.exp(-1j*2*np.pi/N*i) # Used for X[2k+1]
        y5 = np.zeros(int(N/4), dtype=np.complex)
        for i in range(0, int(N/4)):
            y5[i] = (y4[i] + y4[i + int(N/4)]) # Used for X[4k + 1]
        y6 = np.zeros(int(N/8), dtype=np.complex)
        for i in range(0, int(N/8)):
            y6[i] = (y5[i] + y5[i + int(N/8)]) # For X[8k + 1]

        yk_1 = np.fft.fft(y3)
        yk_2 = np.fft.fft(y6)
        yk_1 = np.fft.fftshift(yk_1)
        yk_2 = np.fft.fftshift(yk_2)


        X_k = np.zeros(N, dtype=np.complex)
        for i in range(0,int(N/8)):
            X_k[8*i] = 0
            X_k[8*i + 1] = yk_2[i]
            X_k[8*i + 2] = 0
            X_k[8*i + 3] = 0
            X_k[8*i + 4] = 0
            X_k[8*i + 5] = 0
            X_k[8*i + 6] = 0
            X_k[8*i + 7] = yk_1[i]

        X_k_shift = np.fft.fftshift(X_k)
        x_n = np.fft.ifft(X_k_shift)
        plt.figure(3)
        t = np.arange(0, N, 1);
        plt.plot(t, np.real(x_n), 'r', label='Real')
        plt.plot(t, np.imag(x_n), 'b', label='Imag')
        plt.legend()
        plt.title('Filter Time Domain Sequence')

        plt.figure(4)
        nyquist = int(fs / 2)
        res = 1000
        X_k = X_k/N
        f = np.arange(-nyquist, nyquist, res)
        plt.stem(f, np.real(X_k), 'r', label='Real')
        plt.stem(f, np.imag(X_k), 'b', label='Imag')
        plt.title('Filter Freq. Domain Sequence')
        axis_elem = len(QED_frequencies) - 1
        xmin = 2 * QED_frequencies[0] - 1000
        xmax = 2 * QED_frequencies[axis_elem] + 1000
        plt.axis([xmin, xmax, -5, 5])
        plt.legend()
        plt.show()

    else:
        raise Exception('Filters only defined for 2 qubit system')

    return X_k