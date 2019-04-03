#!/usr/bin/env python3
# Output a THX-like sound effect.  Waaaay too slow but fun to write
import numba
from numba import njit, jit
import struct
import random
import math
import sys
import scipy.io.wavfile as wavfile
import numpy as np

@jit
def key(freq): # Convert a frequency to a piano key
    return 12 * math.log2(freq / 440) + 49

@jit
def freq(key): # Convert a piano key to a frequency
    return math.pow(2, (key-49)/12) * 440


####################PARAMETERS####################

# Target frequencies of the voices
#               C# C# A D A (c) D A D A D F#
# voice_targets = [34, 39, 46, 50, 57, 62, 69]
voice_targets = []
for target in [5, 17, 25, 30, 37, 42, 49, 54, 61, 66, 70]:
#for target in [5, 17] + [1+12*(i//2) + (5 if i%2==0 else 12) for i in range(3,19)]:
#for target in [1+12*(i//2) + (5 if i%2==0 else 12) for i in range(-2,19)]:
    voice_targets.append(target-random.random() * 0.07)
    voice_targets.append(target)
    voice_targets.append(target+random.random() * 0.07)

voice_targets = np.asarray(voice_targets, dtype=np.float32)
# Sample rate
rate = 44100

# Output bounds
output_min = -1.0
output_max = 1.0

# During the "wandering" period, the min/max freqency to wander to
wander_min = 200
wander_max = 400

# Inital frequencies of the voices
voices = np.asarray([random.randint(int(key(wander_min)), int(key(wander_max))) for i in range(len(voice_targets))], dtype=np.float32)

# Random step size, in piano keys, during the wandering period
step = 0.015


# Length in seconds of each of the stages
wander_time = 3
stabilize_time = 3
hold_time = 7
decrescendo_time = 5

####################SIM####################
@njit
def simulate(voices):
    t_max = (wander_time + stabilize_time + hold_time + decrescendo_time) * rate
    crescendo_max = (wander_time + stabilize_time) * rate
    # Output sample buffer (unscaled)
    output = np.zeros((t_max), dtype=np.float32)
    # phase = [0 for _ in voices]
    phase = np.zeros(len(voices))

    # Wandering phase
    for t in range(1, wander_time * rate):
        amplitude = t / crescendo_max
        # amplitude = 1
        sample = 0
        for j in range(voices.shape[0]):
            voice = voices[j]
            old_voice = voice
            voice += random.randint(-1, 1) * step
            new_freq = freq(voice)
            if new_freq < wander_min or new_freq > wander_max:
                voice = old_voice # revert
            voices[j] = voice
            # phase change
            phase[j] += 2 * math.pi * new_freq / rate
            phase[j] %= 2 * math.pi
            # Determine sample based on t and frequency
            voice_out = amplitude * math.sin(phase[j])
            # add to running total
            sample += voice_out
        # output.append(sample)
        output[t] = sample

    # Stabilize phase
    for t in range(wander_time * rate, crescendo_max):
        amplitude = t / crescendo_max
        sample = 0
        for j in range(voices.shape[0]):
            voice = voices[j]
            # Calculate time until done
            target = voice_targets[j]
            slope = (target - voice) / ((wander_time + stabilize_time)*rate - t)
            # Change voice by correct amount
            voice += slope
            voices[j] = voice
            # do the thing with phase and sin
            new_freq = freq(voice)
            phase[j] += 2 * math.pi * new_freq / rate
            phase[j] %= 2 * math.pi
            sample += amplitude * math.sin(phase[j])
        output[t] = sample

    # Hold phase
    for t in range(crescendo_max, crescendo_max + hold_time * rate):
        amplitude = 1.0
        sample = 0
        for j in range(voices.shape[0]):
            voice = voices[j]
            # just do the phase thing!
            new_freq = freq(voice)
            phase[j] += 2 * math.pi * new_freq / rate
            phase[j] %= 2 * math.pi
            sample += amplitude * math.sin(phase[j])
        output[t] = sample

    # Decrescendo
    for t in range(crescendo_max + hold_time * rate, t_max):
        amplitude = (t_max - t) / (t_max - (crescendo_max + hold_time * rate)) 
        sample = 0
        for j,voice in enumerate(voices):
            # just do the phase thing!
            new_freq = freq(voice)
            phase[j] += 2 * math.pi * new_freq / rate
            phase[j] %= 2 * math.pi
            sample += amplitude * math.sin(phase[j])
        output[t] = sample


    ####################OUTPUT####################
    min_sample = output.min()
    max_sample = output.max()

    # Scale output
    for i,sample in enumerate(output):
        output[i] = (sample - min_sample)/(max_sample - min_sample) * (output_max - output_min) + output_min

    return output

if __name__ == '__main__':
    arr = simulate(voices)
    # Write to stdout if we want
    # for sample in arr:
    #     sys.stdout.buffer.write(struct.pack('f', sample))
    wavfile.write('out.wav', rate , arr)
