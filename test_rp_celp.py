#!/usr/bin/env python3

from numpy.testing import assert_almost_equal
from rp_celp import Codec
import numpy as np
import unittest


class TestCodec(unittest.TestCase):
# sine at 440 Hz for sample rate 8000 samp/sec
    sin50 = np.sin(np.linspace(0, 159*2*np.pi/Codec.samp_rate*50, num=160)) * (2**12-1)
    sin220 = np.sin(np.linspace(0, 159*2*np.pi/Codec.samp_rate*220, num=160)) * (2**12-1)
    sin440 = np.sin(np.linspace(0, 159*2*np.pi/Codec.samp_rate*440, num=160)) * (2**12-1)
    sin900 = np.sin(np.linspace(0, 159*2*np.pi/Codec.samp_rate*900, num=160)) * (2**12-1)
    sin1000 = np.sin(np.linspace(0, 159*2*np.pi/Codec.samp_rate*1000, num=160)) * (2**12-1)
    sin2000 = np.sin(np.linspace(0, 159*2*np.pi/Codec.samp_rate*2000, num=160)) * (2**12-1)
    silence = np.array([0., ] * 160)
    silence2 = np.array([0.2 * (2**12-1), ] * 160)

    def test_autocorrelate(self):
        codec = Codec()

        r = codec.autocorrelate(TestCodec.sin440/(2**12-1))
        np.testing.assert_almost_equal(r,
                [ 79.9559, 74.7536, 60.8274, 39.9249, 14.5934, -12.1309,
                    -37.0863, -57.3598, -70.6272, -75.4191, -71.2828],
                decimal=3)

        r = codec.autocorrelate(TestCodec.silence/(2**12-1))
        assert_almost_equal([0., ] * 11, r, decimal=3)

        r = codec.autocorrelate(TestCodec.silence2/(2**12-1))
        assert_almost_equal(r,
                [ 6.4, 6.36, 6.32, 6.28, 6.24, 6.2, 6.16, 6.12, 6.08, 6.04, 6.],
                decimal=3)

    def test_coder_decoder(self):
        codec = Codec()

        r = codec.encode(TestCodec.sin440)
        samples = codec.decode(**r)
        print(samples)

    def test_autocorr2refl_coeffs(self):
        codec = Codec()

# this one produced results out of expected range <-1., 1>
        autocorr = np.array((0.34040, 0.29978, 0.25687, 0.21150, 0.18464,
            0.21603, 0.24005, 0.26593, 0.25947, 0.20285, 0.16447))
        r = codec.autocorr2refl_coeffs(autocorr)
        assert_almost_equal(r, (-0.88066, 0.093427, 0.103595, -0.166237,
            -0.730394, 0.089121, 0.040579, 0.180324, 0.429322, -0.170563),
            decimal=4)

# generated for sin440
        autocorr = np.array((1.34078e+09, 1.25291e+09, 1.01818e+09, 6.66757e+08,
            2.42955e+08, -2.01145e+08, -6.11889e+08, -9.40804e+08, -1.15071e+09,
            -1.21930e+09, -1.14262e+09))
        r = codec.autocorr2refl_coeffs(autocorr)
        assert_almost_equal(r, (-0.93446, 0.89785, 0.425754, 0.215972, 0.111795,
            0.049662, 0.022814, 0.029647, 0.024259, 0.059710),
            decimal=4)

# total silence
        autocorr = np.array((0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.))
        r = codec.autocorr2refl_coeffs(autocorr)
        assert_almost_equal(r, (-0., -0., -0., -0., -0., -0., -0., -0., -0., -0.),
                decimal=4)

# silence with offset
        autocorr = np.array((1.07321e+08, 1.06597e+08, 1.05789e+08, 1.04877e+08,
            1.03885e+08, 1.02803e+08, 1.01634e+08, 1.00378e+08, 9.90601e+07,
            9.76487e+07, 9.61770e+07))
        r = codec.autocorr2refl_coeffs(autocorr)
        assert_almost_equal(r, (-0.993253, 0.061591, 0.072767, 0.052376, 0.05749036,
            0.053660, 0.051856, 0.032840, 0.055199, 0.03077919),
            decimal=4)

    def test_lars2refl_coef(self):
        codec = Codec(approx=True)
        refl_coefs = np.array((-0.98, -0.9, -0.2, 0., 0.6, 0.99))
        lars = codec.refl_coefs2lars(refl_coefs)
        r = codec.lar2refl_coef(lars)
        assert_almost_equal(refl_coefs, r)

        codec = Codec(approx=False)
        lars = codec.refl_coefs2lars(refl_coefs)
        r = codec.lar2refl_coef(lars)
        assert_almost_equal(refl_coefs, r)

    def test_sort_term_filtering(self):
        codec = Codec(approx=False)

        samples = TestCodec.sin1000 / (2**12 - 1)
        autocorr = codec.autocorrelate(samples)
        refl_coefs = codec.autocorr2refl_coeffs(autocorr)
        TestCodec._csv(refl_coefs, name='refl_coefs')
        TestCodec._csv(samples, name='samples')
        samples_ = np.concatenate( ((0., ), samples) )
        refl_coefs = (refl_coefs, refl_coefs, refl_coefs)
        r = codec.short_term_analysis_filtering(samples_, refl_coefs)
        TestCodec._csv(r, name='analysis')
        r2 = codec.short_term_synthesis_filtering(r, refl_coefs)
        TestCodec._csv(r2, name='synthesis')

    @staticmethod
    def _csv(row, name=None):
        if name is not None:
            print("%s," % name, end='')
        for i in row:
            print("\"%s\"," % str(i).replace('.', ','), end='')
        print()


if __name__ == '__main__':
    unittest.main()
