import numpy as np
from bisect import bisect_left


class Codec:
    samp_rate = 8000

    band_expansion = np.array((1., 0.9995, 0.9982, 0.9959, 0.9928, 0.9888,
        0.9839, 0.9781, 0.9716, 0.9641, 0.9559, ) )

# LAR index to quantized value conversion
# TODO: find out and use conversion for TETRAPOL CODEC
    LAR_idx = (
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            )

    def __init__(self, approx=True):
        self.old_lars = None
        self.approx = approx

    def decode(self, lar_idx):
        """Get encoded frame, return 160 samples in 13 bit unifor format
            (16 bit signed int) of decoded audio at rate 8ksampl/sec."""
        lar_quant = self.lar_idxs2lars(lar_idx)
        lars3 = self.lar_interpolate(lar_quant)
        refl_coefs3 = [self.lar2refl_coef(lars) for lars in lars3]

        return refl_coefs3

    def encode(self, samples):
        """Takes 160 samples in 13 bit uniform format (16 bit signed int)
        and compress it using RP-CELP codec."""
# normalize samples to range <-1, 1>
        samples = np.array(samples)/(2**12-1)
        lp = self.lp_analysis(samples)

        return {
                'lar_idx': lp['lar_idx'],
                }

    def lp_analysis(self, samples):
        autocorr = self.autocorrelate(samples)
        autocorr = autocorr * Codec.band_expansion
        refl_coefs = self.autocorr2refl_coeffs(autocorr)
        lars = self.refl_coefs2lars(refl_coefs)
        lar_idx = self.lars2lar_idxs(lars)
        lar_quant = self.lar_idxs2lars(lar_idx)
        lars3 = self.lar_interpolate(lar_quant)
        refl_coefs = [self.lar2refl_coef(lars) for lars in lars3]

        return {
                'lar_idx': lar_idx,
                }

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

    def refl_coefs2lars(self, refl_coefs):
        """If approx is True use approximation as specified in standard,
        if set to False use regular equation."""
        if self.approx:
            return self.refl_coefs2lars_approx(refl_coefs)
        else:
            return self.refl_coefs2lars_eval(refl_coefs)

    def refl_coefs2lars_approx(self, refl_coefs):
        lars = []
        for i in range(len(refl_coefs)):
            refl_c = refl_coefs[i]
            abs_arefl_c = abs(refl_c)
            if abs_arefl_c < 0.675:
                lar = refl_c
            elif abs_arefl_c < 0.950:
                lar = np.copysign(2*abs_arefl_c - 0.675, refl_c)
            else:
                lar = np.copysign(8*abs_arefl_c - 6.375, refl_c)
            lars.append(lar)

        return lars

    def refl_coefs2lars_eval(self, refl_coefs):
        return np.log10((1 + refl_coefs)/(1 - refl_coefs))

    def lars2lar_idxs(self, lars):
        """Return LARs indexes of LARs"""
        lar_idx = []
        for i in range(len(self.LAR_idx)):
            lar = lars[i]
            LAR_idx = self.LAR_idx[i]
            if lar >= LAR_idx[-1]:
                lar_idx.append(len(LAR_idx) - 1)
                continue
            if lar <= LAR_idx[0]:
                lar_idx.append(0)
                continue
            idx = bisect_left(LAR_idx, lar)
            if (LAR_idx[idx-1] - lar) / (LAR_idx[idx] - LAR_idx[idx-1]) >= 0.5:
                lar_idx.append(idx+1)
                continue
            lar_idx.append(idx)

        return lar_idx

    def lar_idxs2lars(self, lar_idx):
        """Return quantized values for LAR indexes."""
        return np.array([self.LAR_idx[i][lar_idx[i]] for i in range(len(lar_idx))])

    def lar_interpolate(self, lars):
        """Create 3 set of LARs interpolating the current and previous set.
        If no previous set is available returns original LARs."""
        if self.old_lars is None:
            self.old_lars = lars
            return (lars, lars, lars)
        lars1 = 0.875*self.old_lars + 0.125*lars
        lars2 = 0.500*self.old_lars + 0.500*lars
        lars3 = 0.125*self.old_lars + 0.875*lars
        self.old_lars = lars
        return (lars1, lars2, lars3)

    def lar2refl_coef(self, lars):
        if self.approx:
            return self.lar2refl_coef_approx(lars)
        else:
            return self.lar2refl_coef_eval(lars)

    def lar2refl_coef_approx(self, lars):
        refl_coefs = []
        for lar in lars:
            abs_lar = abs(lar)
            if abs_lar < 0.675:
                refl_coef = lar
            elif abs_lar < 1.225:
                refl_coef = np.copysign(0.5*abs_lar + 0.3375, lar)
            else:
                refl_coef = np.copysign(0.125*abs_lar + 0.796875, lar)

            refl_coefs.append(refl_coef)
        return refl_coefs

    def lar2refl_coef_eval(self, lars):
        lars = np.power(10, lars)
        return (lars - 1)/(lars + 1)

