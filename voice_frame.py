#!/usr/bin/env python3

from binascii import unhexlify
import json
import numpy as np
import sys


class VoiceFrame:
    coeffs = {
            'LAR01': (1, 0, 23, 22, 21, 20),
            'LAR02': (2, 27, 26, 25, 24),
            'LAR03': (3, 30, 29, 28),
            'LAR04': (4, 31, 33, 32),
            'LAR05': (5, 36, 35, 34),
            'LAR06': (39, 38, 37),
            'LAR07': (42, 41, 40),
            'LAR08': (45, 44, 43),
            'LAR09': (47, 46, 48),
            'LAR10': (51, 50, 49),
            'stochastic_gain1': (8, 63, 62, 61, 64),
            'stochastic_gain2': (13, 87, 86, 85, 84),
            'stochastic_gain3': (18, 109, 108, 107, 106),
            }

    def __init__(self, frame):
        """frame should be byte array with frame data."""
        self.data = []
        for b in frame:
            for i in range(8):
                self.data.append(b & 1)
                b >>= 1

        self.values = {}
        for name, bits in VoiceFrame.coeffs.items():
            value = ''
            for bit in bits:
                value += str(self.data[bit])
            self.values[name] = int(value, 2)

    def get_lars(self):
        d = {}
        for k in self.values:
            if not k.startswith('LAR'):
                continue
            d[k] = self.values[k]
        return d


def get_voice_frame(json_row):
    """Return VOICE frame data or None."""
    j = json.loads(json_row)
    if j['event'] != 'frame':
        return
    j = j['frame']
    if j['type'] != 'VOICE':
        return
    j = j['data']
    if j['encoding'] != 'hex':
        return
    return unhexlify(j['value'])


def print_cvs(items, print_head=False):
    if isinstance(items, dict):
        keys = list(items.keys())
        keys.sort()
        if print_head:
            for i in keys:
                print("%s," % i, end='')
            print()
        for i in keys:
            print("%s," % items[i], end='')
    else:
        for i in items:
            print("%s," % i, end='')
    print()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        in_file = sys.stdin
    elif len(sys.argv) == 2:
        in_file = open(sys.argv[1], 'rt')
    else:
        raise ValueError('Invalid parameters')

    first = True
    for line in in_file:
        frame = get_voice_frame(line)
        if not frame:
            continue

        frame = VoiceFrame(frame)
        lars = frame.get_lars()
        print_cvs(lars, first)
        first = False

