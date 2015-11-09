import numpy as np


class Codec:
    samp_rate = 8000

    band_expansion = np.array(
            (1., 0.9995, 0.9982, 0.9959, 0.9928, 0.9888,
                0.9839, 0.9781, 0.9716, 0.9641, 0.9559, ) )

    def __init__(self):
        pass

    def encode(self, data):
        """Takes 160 samples in 13 bit uniform format (16 bit signed int)
        and compress it using RP-CELP codec."""
        lp = self.lp_analysis(data)

    def lp_analysis(self, data):
        data = np.array(data, dtype=np.int16)
        autocorr = self.autocorrelate(data)
        autocorr = autocorr * Codec.band_expansion

# reflection coefficients, e1 and k0 are index offsets to element with index 0
# e[0][e1] equals in math script e(0, 0)
        p = len(autocorr) - 1
        e1 = p - 1
        e = np.empty( (p, 2*p,) )
        for i in range(p):
            e[0][e1+i] = autocorr[i]
            e[0][e1-i] = autocorr[i]
        e[0][e1 + p] = autocorr[p]

        refl_coef = np.empty(p)
        k0 = - 1
        for i in range(1, p+1):
            refl_coef[k0 + i] = (-e[i - 1][e1 + i])/e[i - 1][e1]
            if i == 10:
                break
            for j in range(-p+1+i, p+1):
                e[i][e1 + j] = e[i-1][e1 + j] + refl_coef[k0 + i] * e[i-1][e1 + i - j]

        return refl_coef

    def autocorrelate(self, data):
        # FIXME: should we autocorrelate using previous sample?
        data10 = np.concatenate((data, np.zeros(10, dtype=np.int16), ))
        autocorr = []
        for i in range(11):
            autocorr.append(np.corrcoef(data, data10[i:i+160])[1][0])
        return autocorr

