import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Radar Parameters ----------
c = 3e8                      # speed of light (m/s)
fc = 77e9                    # carrier freq (Hz)
lam = c / fc                 # wavelength (m)

B = 200e6                    # sweep bandwidth (Hz)
T_chirp = 20e-6              # chirp duration (s)
k = B / T_chirp              # chirp slope (Hz/s)

Fs = 2e6                     # sampling frequency (Hz)
Ns = int(Fs * T_chirp)       # samples per chirp
Nc = 128                     # number of chirps

print("Samples per chirp:", Ns)

# Time axis for one chirp (fast-time)
t = np.arange(Ns) / Fs       # shape: (Ns, )

# ---------- 2. Target Parameters ----------
R_tgt = 50.0                 # target range (m)
v_tgt = 30.0                 # target velocity (m/s)

tau = 2 * R_tgt / c          # time delay (s)
N_delay = int(np.round(tau * Fs))  # delay in samples (integer)

f_D = 2 * v_tgt / lam        # Doppler frequency (Hz)

print("Delay samples:", N_delay, "Doppler freq:", f_D, "Hz")

# ---------- 3. Generate Transmit Signal ----------
# Baseband LFM chirp: s(t) = exp(j * 2π * (0.5 * k * t^2))
# We'll ignore carrier term in simulation (simulate baseband)
tx_chirp = np.exp(1j * 2 * np.pi * (0.5 * k * t**2))  # shape: (Ns,)

# Create 2D transmit signal: Nc chirps stacked
tx = np.tile(tx_chirp, (Nc, 1))   # shape: (Nc, Ns)

# ---------- 4. Generate Received Signal ----------
# Start with zeros
rx = np.zeros_like(tx, dtype=complex)

# Only fill for samples after the delay
# We'll apply Doppler as a phase shift per chirp
alpha = 1.0  # reflection amplitude

for m in range(Nc):  # slow-time index (chirp index)
    # Doppler phase for this chirp
    phi_m = 2 * np.pi * f_D * m * T_chirp
    
    # For this chirp, copy delayed version of tx
    if N_delay < Ns:
        rx[m, N_delay:] = alpha * tx[m, :Ns - N_delay] * np.exp(1j * phi_m)

# Add AWGN noise
SNR_dB = 20
signal_power = np.mean(np.abs(rx)**2)
noise_power = signal_power / (10**(SNR_dB / 10))
noise = np.sqrt(noise_power/2) * (np.random.randn(*rx.shape) + 1j*np.random.randn(*rx.shape))
rx_noisy = rx + noise

# ---------- 5. Dechirp / Mix ----------
beat = tx * np.conj(rx_noisy)    # shape: (Nc, Ns)

# ---------- 6. Range FFT (fast-time) ----------
# Apply window (optional)
window_range = np.hanning(Ns)
beat_win = beat * window_range  # broadcast along Ns

# FFT along samples axis=1
range_fft = np.fft.fft(beat_win, axis=1)
range_fft = np.fft.fftshift(range_fft, axes=1)  # shift zero freq to center

# Compute range axis
freqs = np.fft.fftfreq(Ns, d=1/Fs)
freqs_shifted = np.fft.fftshift(freqs)

# Mapping frequency to range for FMCW:
# f_b = (2 * k * R) / c  => R = (c * f_b) / (2 * k)
R_axis = (c * freqs_shifted) / (2 * k)

# Take one chirp's range profile for plotting
range_profile = np.abs(range_fft[0, :])

# ---------- 7. Doppler FFT (slow-time) ----------
# Pick only positive range bins (e.g., mid to end)
# For R-D map, we'll use full 2D FFT: slow-time (Nc) x fast-time (Ns)
# First, maybe keep only a subset of range bins
RD_input = beat[:, :]  # you can also use beat_win

# Windowing in Doppler dimension (optional)
window_dopp = np.hanning(Nc).reshape(Nc, 1)
RD_win = RD_input * window_dopp

# 2D FFT: first over fast-time, then over slow-time
RD = np.fft.fft2(RD_win, s=(Nc, Ns))
RD = np.fft.fftshift(RD, axes=(0,1))

RD_mag = np.abs(RD)

# Doppler axis
fd = np.fft.fftfreq(Nc, d=T_chirp)        # Doppler frequency bins
fd_shifted = np.fft.fftshift(fd)
# Map Doppler freq to velocity: v = fd * lam / 2
v_axis = fd_shifted * lam / 2

# ---------- 8. Plot Results ----------

# 8a. Range profile (one chirp)
plt.figure()
plt.plot(R_axis, range_profile)
plt.xlim(0, 200)               # show up to 200m
plt.xlabel("Range (m)")
plt.ylabel("Amplitude")
plt.title("Range Profile (Chirp 0)")
plt.grid(True)

# 8b. Range-Doppler Map
# Limit range and velocity region for visualization
R_min_plot, R_max_plot = 0, 150
V_min_plot, V_max_plot = -60, 60

# Create meshgrid (for plotting)
R_mesh, V_mesh = np.meshgrid(R_axis, v_axis)

plt.figure()
plt.pcolormesh(R_mesh, V_mesh, 20 * np.log10(RD_mag + 1e-12), shading='auto')
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Range-Doppler Map (dB)")
plt.xlim(R_min_plot, R_max_plot)
plt.ylim(V_min_plot, V_max_plot)
plt.colorbar(label="Magnitude (dB)")
plt.show()
