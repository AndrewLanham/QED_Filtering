# This file simulates the reduced FFT process. Two time-domain signals are generated at sample rate fs containing
# arbitrary frequency information from -31 to 31 kHz at spacings of 2kHz, for a total of 32 frequencies. The length
# of the time-domain signals is equal to one period of the lowest frequency, which for this demonstration is 1KHz. Thus
# the number of samples Ns = 1e-3*f_s
#
# The time domain signal is fed into two separate ``filters" each of which performs a modified FFT that only recovers
# half of total frequency information. This saves (N log(N) / 2) multiply adds per filter, as only half of the FFT
# is computed. The frequency domain data is zero padded in the bins where frequency information was not calculated,
# and a standard IFFT is performed for each of them.
#
# The modified FFT leverages the "Decimation in Frequency" FFT approach to calculate the partial FFT of the time-domain
# data.
#
# This approach requires that Ns is a power of 2. Thus fs/1000 must be an integer power of two.
import matplotlib.pyplot as plt
import SynthesizeSignals as syn
import BasisFunctions as bs
import Filter as fft
import numpy as np

def main():
    fs = 256000
    f_min = 1000
    QED_frequencies = np.arange(-3000,5000,2000) # Specify operating frequencies
    num_qubits = np.log2(len(QED_frequencies))
    (x_n,x_k) = syn.SynthesizeSignals(fs, QED_frequencies) # This function is used to simulate one period of a QED input signal.
                                          # The phase is random.


    # From this point onward we attempt to simulate projective filtering using the time domain signal x_n
    re_psi = np.real(x_n)
    im_psi = np.imag(x_n) # Each filter has access to the real and imaginary signals separately

    # We simulate two filters FilterProj0 and FilterProj1. The configuration is decided based on the target qubit where
    # qubit 0 is the least significant qubit and implies we should filter every other frequency and qubit n is the most
    # significant qubit
    target_qubit = 0 # choose target qubit here ( 0 or 1)

    # Project onto qubit n state 0 or 1
    p_n_0_prefilter = bs.BasisFunctionProj0(x_n, fs, QED_frequencies, target_qubit) # Apply the basis function for filter zero based on the target qubit index and return it
    p_n_1_prefilter = bs.BasisFunctionProj1(x_n, fs, QED_frequencies, target_qubit) # Apply the basis function for filter one based on the target qubit index and return it

    # Filter the signals
    p_n_0 = fft.DIF_FFT(target_qubit, p_n_0_prefilter,fs, QED_frequencies)
    p_n_1 = fft.DIF_FFT(target_qubit, p_n_1_prefilter,fs, QED_frequencies)

    

if __name__ == "__main__":
    main()