# This file contains functions related to the projection operations
import numpy as np
import matplotlib.pyplot as plt

# Performs the projection onto the zero state based on the qubit index idx.
# The important thing about this function is that the time domain signal x_n is
# modulated by some complex exponential
def BasisFunctionProj0(x_n,fs,QED_frequencies,target_qubit_idx):
    N  = len(x_n)
    t = np.linspace(0,0.001,N) # 1ms time axis
    f_0 = 1*(2**target_qubit_idx)*1000 # = 2*pi*1000 for qubit 0, 2*pi*2000 for qubit 1, 2*pi*4000 for qubit 2, and so on
    b_n = np.exp(1j*2*np.pi*f_0*t)
    conj_b_n = np.conj(b_n)


    plt.figure(6)
    t = np.arange(0, N, 1);
    plt.plot(t, np.real(b_n), 'r', label='Real')
    plt.plot(t, np.imag(b_n), 'b', label='Imag')
    plt.legend()
    plt.title('Time Domain Sequence Basis0')


    plt.figure(5)
    nyquist = int(fs / 2)
    res = 1000
    b_k = np.fft.fft(b_n)
    b_k = (np.fft.fftshift(b_k))/N
    f = np.arange(-nyquist, nyquist, res)
    plt.stem(f, np.real(b_k), 'r', label='Real')
    plt.stem(f, np.imag(b_k), 'b', label='Imag')
    plt.title('Frequency Domain Sequence Basis0')
    axis_elem = len(QED_frequencies) - 1
    xmin = 2 * QED_frequencies[0] - 1000
    xmax = 2 * QED_frequencies[axis_elem] + 1000
    plt.axis([xmin, xmax, -5, 5])
    plt.legend()
    plt.show()

    p_n = conj_b_n*x_n # Performs the projection

    # Plotting parameters
    plt.figure(3)
    t = np.arange(0, N, 1);
    plt.plot(t, np.real(p_n), 'r', label='Real')
    plt.plot(t, np.imag(p_n), 'b', label='Imag')
    plt.legend()
    plt.title('Time Domain Sequence Proj0')

    plt.figure(4)
    nyquist = int(fs / 2)
    res = 1000
    p_k = np.fft.fft(p_n)
    p_k = (np.fft.fftshift(p_k))/N
    f = np.arange(-nyquist, nyquist, res)
    plt.stem(f, np.real(p_k), 'r', label='Real')
    plt.stem(f, np.imag(p_k), 'b', label='Imag')
    plt.title('Frequency Domain Sequence Proj0')
    axis_elem = len(QED_frequencies) - 1
    xmin = 2*QED_frequencies[0] - 1000
    xmax = 2*QED_frequencies[axis_elem] + 1000
    plt.axis([xmin, xmax, -5, 5])
    plt.legend()
    plt.show()

    return p_n




# Performs the projection onto the one state based on the qubit index idx.
# The important thing about this function is that the time domain signal x_n is
# modulated by some complex exponential
def BasisFunctionProj1(x_n,fs,QED_frequencies,target_qubit_idx):
    N = len(x_n)
    t = np.linspace(0, 0.001, N)  # 1ms time axis
    f_0 = -1 * (2 ** target_qubit_idx) * 1000  # = 1000 for qubit 0, 2000 for qubit 1, 4000 for qubit 2, and so on
    b_n = np.exp(1j * 2 * np.pi * f_0 * t)
    conj_b_n = np.conj(b_n) # Complex conjugate required for projection

    plt.figure(6)
    t = np.arange(0, N, 1);
    plt.plot(t, np.real(b_n), 'r', label='Real')
    plt.plot(t, np.imag(b_n), 'b', label='Imag')
    plt.legend()
    plt.title('Time Domain Sequence Basis1')


    plt.figure(5)
    nyquist = int(fs / 2)
    res = 1000
    b_k = np.fft.fft(b_n)
    b_k = (np.fft.fftshift(b_k))/N
    f = np.arange(-nyquist, nyquist, res)
    plt.stem(f, np.real(b_k), 'r', label='Real')
    plt.stem(f, np.imag(b_k), 'b', label='Imag')
    plt.title('Frequency Domain Sequence Basis1')
    axis_elem = len(QED_frequencies) - 1
    xmin = 2 * QED_frequencies[0] - 1000
    xmax = 2 * QED_frequencies[axis_elem] + 1000
    plt.axis([xmin, xmax, -5, 5])
    plt.legend()
    plt.show()

    p_n = conj_b_n * x_n  # Performs the projection

    # Plotting parameters
    plt.figure(3)
    t = np.arange(0, N, 1);
    plt.plot(t, np.real(p_n), 'r', label='Real')
    plt.plot(t, np.imag(p_n), 'b', label='Imag')
    plt.legend()
    plt.title('Time Domain Sequence Proj1')

    plt.figure(4)
    nyquist = int(fs / 2)
    res = 1000
    p_k = np.fft.fft(p_n)
    p_k = (np.fft.fftshift(p_k)/N)
    f = np.arange(-nyquist, nyquist, res)
    plt.stem(f, np.real(p_k), 'r', label='Real')
    plt.stem(f, np.imag(p_k), 'b', label='Imag')
    plt.title('Frequency Domain Sequence Proj1')
    axis_elem = len(QED_frequencies) - 1
    xmin = 2 * QED_frequencies[0] - 1000
    xmax = 2 * QED_frequencies[axis_elem] + 1000
    plt.axis([xmin, xmax, -5, 5])
    plt.legend()
    plt.show()

    return p_n
