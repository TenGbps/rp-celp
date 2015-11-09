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
        return autocorr

    def autocorrelate(self, samples):
        # TODO: should we autocorrelate using samples from previous frame?
        return [ np.correlate(samples, samples[i:])[0] for i in range(11) ]

