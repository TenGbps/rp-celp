from rp_celp import Codec
import numpy as np
import unittest


class TestCodec(unittest.TestCase):
# sine at 440 Hz for sample rate 8000 samp/sec
    sin440 = np.array( [ np.sin(x*2*np.pi/Codec.samp_rate*440) * (2**12-1)
        for x in range(160) ], dtype=np.int16)

    def test_coder(self):
        codec = Codec()
        codec.encode(TestCodec.sin440)


if __name__ == '__main__':
    unittest.main()
