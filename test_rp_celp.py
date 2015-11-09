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


if __name__ == '__main__':
    unittest.main()
