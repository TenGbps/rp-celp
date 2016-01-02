#!/usr/bin/env python3

from binascii import unhexlify
from rp_celp import Codec
from struct import pack
from voice_frame import VoiceFrame
import json
import sys
import wave


class VoiceDecoder:
    """Decode VOICE frames from JSON format into WAV file."""
    def __init__(self, out_file):
        self.out_file = wave.open(out_file, 'wb')
        self.out_file.setnchannels(1)
        self.out_file.setsampwidth(2)
        self.out_file.setframerate(8000)
        self.codec = Codec(approx=False)

    def decode_frame(self, frame):
        data = self.get_voice_data(frame)
        if data is None:
            return
        voice_frame = VoiceFrame(data)
        lar_idx = []
        for i in range(1, 11):
            lar_idx.append(voice_frame.values['LAR%02d' % i])
        subframe1 = { 'stochastic_gain': voice_frame.values['stochastic_gain1'] }
        subframe2 = { 'stochastic_gain': voice_frame.values['stochastic_gain2'] }
        subframe3 = { 'stochastic_gain': voice_frame.values['stochastic_gain3'] }
        snd = self.codec.decode(lar_idx=lar_idx, subframe1=subframe1,
                subframe2 = subframe2, subframe3=subframe3)
        snd = snd*32767
        snd = [int(s) for s in snd]
        snd = pack('>160h', *snd)
        self.out_file.writeframes(snd)

    def get_voice_data(self, frame):
        """Return VOICE frame data from JSON encoded frame or None."""
        j = json.loads(frame)
        if j['event'] != 'frame':
            return
        j = j['frame']
        if j['type'] != 'VOICE':
            return
        j = j['data']
        if j['encoding'] != 'hex':
            return
        return unhexlify(j['value'])


if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    voice_decoder = VoiceDecoder(out_file)
    in_file = open(in_file, 'rt')
    for j in in_file:
        voice_decoder.decode_frame(j)

    del voice_decoder

