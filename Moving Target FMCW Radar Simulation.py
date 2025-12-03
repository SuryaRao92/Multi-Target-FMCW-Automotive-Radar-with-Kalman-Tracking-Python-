import numpy as np
import matplotlib.pyplot as plt

# ---------------- Radar Parameters ----------------
c = 3e8                       # speed of light
fc = 77e9                     # carrier frequency
lam = c / fc
B = 200e6                     # bandwidth
T_chirp = 20e-6               # chirp time
k = B / T_chirp               # chirp slope
Fs = 2e6                      # sampling frequency
Ns = int(Fs * T_chirp)        # samples per chirp
Nc = 128                      # number of chirps

# ---------------- Moving Target Parameters ----------------
R0 = 60.0       # starting range (meters)
v = -5.0        # velocity m/s (- means moving toward radar)

ranges_detected = []

for chirp in range(Nc):

    # Calculate target position for current chirp
    R_tgt = R0 + v * chirp * T_chirp
    ranges_detected.append(R_tgt)

    # Time delay based on distance
    tau = 2 * R_tgt / c
    N_delay = int(tau * Fs)

    # Generate Tx chirp
    t = np.arange(Ns) / Fs
    tx = np.exp(1j * 2 * np.pi * (0.5 * k * t**2))

    # Create Rx signal (delayed copy + doppler shift)
    fd = 2 * v / lam
    doppler_phase = np.exp(1j * 2 * np.pi * fd * chirp * T_chirp)

    rx = np.zeros(Ns, dtype=complex)
    if N_delay < Ns:
        rx[N_delay:] = tx[:Ns - N_delay] * doppler_phase

    # Mix TX & RX
    beat = tx * np.conjugate(rx)

    # Range FFT
    range_fft = np.fft.fft(beat)
    freqs = np.fft.fftfreq(Ns, 1/Fs)
    R_vals = (c * np.abs(freqs)) / (2 * k)

    peak_index = np.argmax(np.abs(range_fft))
    detected_range = R_vals[peak_index]
    ranges_detected[-1] = detected_range

# ---------------- PLOT Range Tracking ----------------
plt.figure(figsize=(8, 4))
plt.plot(ranges_detected, marker='o')
plt.title("Moving Target Range Tracking Over Time")
plt.xlabel("Chirp Number (Time)")
plt.ylabel("Detected Range (m)")
plt.grid(True)
plt.show()
