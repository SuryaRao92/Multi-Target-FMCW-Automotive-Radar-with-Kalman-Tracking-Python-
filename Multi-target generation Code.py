import numpy as np
import matplotlib.pyplot as plt

# Radar Parameters
c = 3e8
fc = 77e9
lam = c / fc
B = 200e6
T_chirp = 20e-6
Fs = 2e6
Ns = int(T_chirp * Fs)
Nc = 128
k = B / T_chirp

# Target Scenario
targets = [
    {"R": 80, "v": -25},   # Car A
    {"R": 120, "v": -10},  # Car B
    {"R": 60, "v": 15},    # Bike
    {"R": 40, "v": 2}      # Pedestrian
]

t = np.arange(Ns) / Fs
tx = np.exp(1j * 2 * np.pi * (0.5 * k * t**2))

all_chirps = []

for m in range(Nc):  # For each chirp
    rx = np.zeros(Ns, dtype=complex)

    for tgt in targets:
        # Update target movement
        tgt["R"] = tgt["R"] + tgt["v"] * T_chirp

        tau = 2 * tgt["R"] / c
        N_delay = int(tau * Fs)

        if N_delay < Ns:
            fd = 2 * tgt["v"] / lam
            doppler = np.exp(1j * 2 * np.pi * fd * m * T_chirp)

            rx[N_delay:] += tx[:Ns - N_delay] * doppler

    # add noise
    rx += (np.random.randn(Ns) + 1j * np.random.randn(Ns)) * 0.01
    beat = tx * np.conjugate(rx)
    all_chirps.append(beat)

print("Multi-target beat signal generated successfully!")
