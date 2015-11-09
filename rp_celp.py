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
        lp = self.lp_analysis()

    def lp_analysis(self, data):
        data = np.array(data, dtype=np.int16)
        autocorr = self.autocorrelate(data)
        autocorr = autocorr * Codec.band_expansion
        return autocorr

    def autocorrelate(self, data):
        # FIXME: should we autocorrelate using previous sample?
        data10 = np.concatenate((data, np.zeros(10, dtype=np.int16), ))
        autocorr = []
        for i in range(11):
            autocorr.append(np.corrcoef(data, data10[i:i+160])[1][0])
        return autocorr

