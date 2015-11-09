import numpy as np


class Codec:
    samp_rate = 8000

    band_expansion = np.array((1., 0.9995, 0.9982, 0.9959, 0.9928, 0.9888,
        0.9839, 0.9781, 0.9716, 0.9641, 0.9559, ) )

    def __init__(self):
        pass

    def encode(self, samples):
        """Takes 160 samples in 13 bit uniform format (16 bit signed int)
        and compress it using RP-CELP codec."""
# normalize samples to range <-1, 1>
        samples = np.array(samples)/(2**12-1)
        lp = self.lp_analysis(samples)
        return lp

    def lp_analysis(self, samples):
        autocorr = self.autocorrelate(samples)
        autocorr = autocorr * Codec.band_expansion
        refl_coefs = self.autocorr2refl_coeffs(autocorr)
        return refl_coefs

    def autocorrelate(self, samples):
        # TODO: should we autocorrelate using samples from previous frame?
        return [ np.correlate(samples, samples[i:])[0] for i in range(11) ]

    def autocorr2refl_coeffs(self, autocorr):
        """Literature: Speech Coding Algorithms, Wai C. Chu page 119."""
# prevent divission by zero bellow
        if abs(autocorr[0]) < 2**-12:
            autocorr[0] = 2**-12

# e1 and r0 are index offsets to element with math index 0
# example: math e(0, -1) => python e[0][e1 - 1]
        M = len(autocorr) - 1
        e1 = M - 1
        e = np.empty( (M, 2*M,) )
# e[0] is simetricaly filled around element 0 with autocorr
        e[0][e1:] = autocorr
        e[0][:e1] = autocorr[-2:0:-1]

        refl_coefs = np.empty(M)
        r0 = - 1
        for l in range(1, M+1):
            refl_coefs[r0+l] = (-e[l-1][e1+l])/e[l-1][e1]
# TODO: clip values in range, in some cases they go a bit out of bounds (check!)
            refl_coefs[r0+l] = np.clip(refl_coefs[r0+l], -0.99999, 0.99999)
            if l == M:
                break
            for k in range(-M+1+l, M+1):
                e[l][e1+k] = e[l-1][e1+k] - refl_coefs[r0+l] * e[l-1][e1+l-k]

        return refl_coefs

