import numpy as np
import scipy.signal
from bisect import bisect_left


class Codec:
    samp_rate = 8000

    band_expansion = np.array((1., 0.9995, 0.9982, 0.9959, 0.9928, 0.9888,
        0.9839, 0.9781, 0.9716, 0.9641, 0.9559, ) )

# LAR index to quantized value conversion
# TODO: find out and use conversion for TETRAPOL CODEC
    LAR_idx = (
            np.linspace(-2.0643047, 0.38465041, 64),
            np.linspace(-0.7255134, 2.02982235, 32),
            np.linspace(-1.3538182, 0.62888535, 16),
            np.linspace(-0.4090979, 1.36354402, 16),
            np.linspace(-0.7099015, 0.57737373, 16),
            np.linspace(-0.4737691, 0.82117651, 8),
            np.linspace(-0.8141674, 0.47146487, 8),
            np.linspace(-0.5001262, 0.71935197, 8),
            np.linspace(-0.7204825, 0.47273149, 8),
            np.linspace(-0.3206912, 0.55081164, 8),
            )

    # Quantization Gain values for indexes 0..31
    QLBG = (0, 5, 11, 19, 27, 35, 43, 51, 59, 71, 87, 103, 119, 143, 175, 207,
            239, 287, 351, 415, 479, 575, 703, 831, 959, 1151, 1407, 1663,
            1919, 2303, 2815, 3583, )
    QLBG_norm = np.array(QLBG) / max(QLBG)

# size of subframes
    N = (56, 48, 56)

    def __init__(self, approx=True):
        self.old_lars = None
        self.approx = approx
# required one extra sample from past
        self.prec = np.zeros(81)
# decoder, keep history
        self.v = np.zeros(11)
# encoder, keep 1 sample from previous frame as initial value for short term filter
        self.s_prev = 0
# white noise, excitation vector for synthesis filter
        self.noise = np.clip(np.random.normal(size=(1600000,)), -2.1, 2.1)
        h = scipy.signal.firwin(numtaps=8, cutoff=3000, nyq=Codec.samp_rate/2)
        self.noise=scipy.signal.lfilter(h, 1.0, self.noise)
        self.noise_offs = 0

    def decode(self, lar_idx, subframe1, subframe2, subframe3):
        """Get encoded frame, return 160 samples in 13 bit unifor format
            (16 bit signed int) of decoded audio at rate 8ksampl/sec."""
        lar_quant = self.lar_idxs2lars(lar_idx)
        lars3 = self.lar_interpolate(lar_quant)
        refl_coefs3 = [self.lar2refl_coef(lars) for lars in lars3]

        self.noise_offs += 160
        if self.noise_offs >= len(self.noise):
            self.noise_offs = 160
        d = self.noise[self.noise_offs-160:self.noise_offs] * 0.00003
        #d[0] = subframe1['stochastic_gain'] / 2.**5
        #d[self.N[1]] = subframe2['stochastic_gain'] / 2.**5
        #d[self.N[1] + self.N[2]] = subframe3['stochastic_gain'] / 2.**5

        s = self.short_term_synthesis_filtering(d, refl_coefs3)
        return s

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
        s = self.win_shift(samples)
        d = self.short_term_analysis_filtering(s, refl_coefs)

        return {
                'lar_idx': lar_idx,
                }

    def autocorrelate(self, samples):
        # TODO: should we autocorrelate using samples from previous frame?
        return [ np.correlate(samples, samples[i:])[0] for i in range(11) ]

    def autocorr2refl_coeffs(self, autocorr):
        """Literature: Speech Coding Algorithms, Wai C. Chu page 119 (we fixed? the sign
        in 4.80. IT++ lerouxguegenrc() (but this function accessed array out of range)."""
# prevent divission by zero bellow
        if abs(autocorr[0]) < 2**-12:
            autocorr[0] = 2**-12

        M = len(autocorr) - 1
        r = np.empty(2*M+1)
        rny = np.empty(2*M+1)
        refl_coefs = np.empty(M)

        for j in range(0, M+1):
            r[M-j] = autocorr[j]
            r[M+j] = autocorr[j]

        for m in range(1, M+1):
            refl_coefs[m-1] = -r[M+m] / r[M]
            if m == M:
                break
            for j in range(-M+m+1, M+1):
                rny[M+j] = r[M+j] + refl_coefs[m-1] * r[M+m-j]
            for j in range(-M, M+1):
                r[M+j] = rny[M+j]

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

    def win_shift(self, samples):
        """5.8 Temporal windows shift.
        Returns current window samples with one extra sample from past at the begin."""
        s = np.empty(161)
        s[:81] = self.prec
        s[81:] = samples[:80]
        self.prec[:] = samples[79:]
        return s

    def short_term_analysis_filtering(self, s, refl_coefs):
        """5.9 Short term analysis filtering."""
        d = np.empty(len(s))
        r = refl_coefs[0]
# k-1 element of tmp1/tmp2 is not defined, we use value from previous frame if possible
        tmp1 = np.empty((11, len(s) + 1))
        tmp2 = np.empty((11, len(s) + 1))
        tmp1[0][:-1] = s
        tmp2[0][:-1] = s
        tmp1[0][-1] = self.s_prev
        tmp2[0][-1] = self.s_prev
        self.s_prev = s[0]
        for k in range(0, len(s)):
            for i in range(1, 11):
                tmp1[i][k] = tmp1[i-1][k] + r[i-1]*tmp2[i-1][k-1]
                tmp2[i][k] = tmp2[i-1][k-1] + r[i-1]*tmp1[i-1][k]
            d[k] = tmp1[10][k]
            if k == self.N[0]:
                r = refl_coefs[1]
            elif k == self.N[0] + self.N[1]:
                r = refl_coefs[2]

        return d

    def short_term_synthesis_filtering(self, d, refl_coefs):
        """6.3 Short term synthesis filter. Original is broken using GSM version."""
        r = refl_coefs[0]
        v = self.v
        s = np.empty(len(d))

        for k in range(len(d)):
            sri = d[k]
            for i in range(1, 11):
                sri = sri - r[10 - i]*v[10-i]
                v[11-i] = v[10-i] + r[10 - i]*sri
            s[k] = sri
            v[0] = sri
            if k == self.N[0]:
                r = refl_coefs[1]
            elif k == self.N[0] + self.N[1]:
                r = refl_coefs[2]

        self.v = v

        return s
