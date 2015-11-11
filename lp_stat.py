#!/usr/bin/env python3

from rp_celp import Codec
from struct import unpack
import numpy as np
import sys


class LPStat:
    def __init__(self, in_file_name):
        """in_file_name should contain 16 bit mono audio with range +-(2**12-1)."""
        codec = Codec()
        n = 1000000
        v = [[], [], [], [], [], [], [], [], [], []]
        with open(in_file_name, 'rb') as f:
            while True:
                n -= 1
                if n <= 0:
                    break
                frame = f.read(2*160)
                if len(frame) != 2*160:
                    break
                frame = unpack('h'*160, frame)
                cframe = codec.encode(frame)
                for i in range(10):
                    v[i].append(cframe[i])

        hists = []
        for vn in v:
            hists.append(np.histogram(vn, bins=32, range=(0, 32), density=False))

        # values in first row
        print(",", end='')
        val = hists[0][1]
        for v in val:
            print("%f," % v, end='')
        print()

        for i in range(len(hists)):
            print("%d," % i, end='')
            cnt = hists[i][0]
            for c in cnt:
                print("%d," % c, end='')
            print()
        print()


if __name__ == '__main__':
    lp_stat = LPStat(sys.argv[1])
