import numpy as np
from bisect import bisect_left


class Codec:
    samp_rate = 8000

    band_expansion = np.array((1., 0.9995, 0.9982, 0.9959, 0.9928, 0.9888,
        0.9839, 0.9781, 0.9716, 0.9641, 0.9559, ) )

# LAR index to quantized value conversion
# TODO: find out and use conversion for TETRAPOL CODEC
    LAR_idx = (
            np.linspace(-0.97, 0.97, 64),
            np.linspace(-0.97, 0.97, 64),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 32),
            np.linspace(-0.97, 0.97, 16),
            np.linspace(-0.97, 0.97, 16),
            np.linspace(-0.97, 0.97, 16),
            np.linspace(-0.97, 0.97, 16),
            np.linspace(-0.97, 0.97, 16),
            )

# size of subframes
    N = (56, 48, 56)

    def __init__(self, approx=True):
        self.old_lars = None
        self.approx = approx
# required one extra sample from past
        self.prec = np.zeros(81)
# decoder, keep history for short term synthesis filter
        self.v = np.zeros((11, 161))
# encoder, keep 1 sample from previous frame as initial value for short term filter
        self.s_prev = 0

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
        """6.3 Short term synthesis filtering."""
        s = np.empty(len(d))
        tmp = np.empty((11, len(d)))
        tmp[0][:] = d
        n = 0
        # use range 1, N instead of 0, N-1
        for k in range(1, len(d)+1):
            for i in range(1, 11):
                tmp[i][k-1] = tmp[i-1][k-1] - refl_coefs[n][10-i]*self.v[10-i][k-1]
                self.v[11-i][k] = self.v[10-i][k-1] - refl_coefs[n][10-i]*tmp[i][k-1]
            if k == self.N[0]:
                n = 1
            elif k == self.N[0] + self.N[1]:
                n = 2
            self.v[0][k] = tmp[10][k-1]
            s[k-1] = tmp[10][k-1]
        # prepare for next round
        for i in range(len(self.v)):
            self.v[i][0] = self.v[i][-1]

        return s

