#!/usr/bin/env python3

from numpy.testing import assert_almost_equal
from rp_celp import Codec
import numpy as np
import unittest


class TestCodec(unittest.TestCase):
# sine at 440 Hz for sample rate 8000 samp/sec
    sin440 = np.sin(np.linspace(0, 159*2*np.pi/Codec.samp_rate*440, num=160)) * (2**12-1)
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

    def test_autocorr2refl_coeffs(self):
        codec = Codec()

# this one produced results out of expected range <-1., 1>
        autocorr = np.array((0.34040, 0.29978, 0.25687, 0.21150, 0.18464,
            0.21603, 0.24005, 0.26593, 0.25947, 0.20285, 0.16447))
        r = codec.autocorr2refl_coeffs(autocorr)
        assert_almost_equal(r, (-0.88066, -0.86178, -0.84593, -0.83941,
            -0.85674, -0.89334, -0.93872, -0.98114, -0.99999, -0.99999),
            decimal=4)

# generated for sin440
        autocorr = np.array((1.34078e+09, 1.25291e+09, 1.01818e+09, 6.66757e+08,
            2.42955e+08, -2.01145e+08, -6.11889e+08, -9.40804e+08, -1.15071e+09,
            -1.21930e+09, -1.14262e+09))
        r = codec.autocorr2refl_coeffs(autocorr)
        assert_almost_equal(r, (-0.93446, -0.87155, -0.80088, -0.71565,
            -0.60413, -0.44283, -0.17919, 0.29794, 0.92724, 0.93862),
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
        assert_almost_equal(r, (-0.99325, -0.99281, -0.99234, -0.99187, -0.99140,
            -0.99094, -0.99047, -0.99000, -0.98953, -0.98906), decimal=4)


if __name__ == '__main__':
    unittest.main()
