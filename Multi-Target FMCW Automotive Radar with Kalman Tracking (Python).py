import numpy as np
import matplotlib.pyplot as plt

# --------------------- RADAR PARAMETERS ---------------------
c = 3e8
fc = 77e9
lam = c / fc
B = 200e6
T_chirp = 20e-6
Fs = 2e6
Ns = int(T_chirp * Fs)
Nc = 128
k = B / T_chirp

# --------------------- MULTI-TARGET SCENARIO ---------------------
targets = [
    {"R": 80, "v": -25},   # Car A - approaching fast
    {"R": 120, "v": -10},  # Car B - approaching slow
    {"R": 60, "v": 15},    # Bike - moving away
    {"R": 40, "v": 2}      # Pedestrian - walking away slowly
]

# Transmit chirp generation
t = np.arange(Ns) / Fs
tx = np.exp(1j * 2 * np.pi * (0.5 * k * t**2))
all_chirps = []

# --------------------- RADAR SIGNAL SIMULATION ---------------------
for m in range(Nc):
    rx = np.zeros(Ns, dtype=complex)

    for tgt in targets:
        tgt["R"] = tgt["R"] + tgt["v"] * T_chirp  # update motion

        tau = 2 * tgt["R"] / c
        N_delay = int(tau * Fs)

        if N_delay < Ns:
            fd = 2 * tgt["v"] / lam
            doppler = np.exp(1j * 2 * np.pi * fd * m * T_chirp)
            rx[N_delay:] += tx[:Ns - N_delay] * doppler

    rx += (np.random.randn(Ns) + 1j*np.random.randn(Ns)) * 0.01  # noise
    beat = tx * np.conjugate(rx)
    all_chirps.append(beat)

print("Beat signal simulation completed with 4 moving targets.")


# --------------------- RANGE FFT & PEAK DETECTION ---------------------
detected_ranges_each_frame = []

for beat in all_chirps:
    R_fft = np.fft.fft(beat)
    freq_axis = np.fft.fftfreq(Ns, 1/Fs)
    R_vals = (c * np.abs(freq_axis)) / (2 * k)

    # get 4 highest peaks
    peak_indices = np.argsort(np.abs(R_fft))[-4:]
    detected_ranges = sorted(R_vals[peak_indices])
    detected_ranges_each_frame.append(detected_ranges)

print("Range FFT and multi-target peak detection completed.")


# --------------------- RANGE-DOPPLER MAP ---------------------
beat_matrix = np.array(all_chirps)

window = np.hanning(Ns)
beat_matrix = beat_matrix * window

range_fft = np.fft.fft(beat_matrix, axis=1)
RD_map = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
RD_mag = 20 * np.log10(np.abs(RD_map) + 1e-12)

freqs = np.fft.fftfreq(Ns, 1/Fs)
R_axis = (c * np.abs(freqs)) / (2 * k)

fd = np.fft.fftfreq(Nc, T_chirp)
v_axis = fd * lam / 2

plt.figure(figsize=(8, 6))
plt.pcolormesh(R_axis, v_axis, RD_mag, shading='auto')
plt.title("Range-Doppler Map (Multiple Targets)")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.colorbar(label="Magnitude (dB)")
plt.xlim(0, 150)
plt.ylim(-60, 60)
plt.show()


# --------------------- KALMAN FILTER CLASS ---------------------
class KalmanFilter:
    def __init__(self, dt, process_noise=1, measurement_noise=8):
        self.dt = dt
        self.x = np.array([[0], [0]])
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.P = np.eye(2)
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x


dt = T_chirp
kalman_filters = [KalmanFilter(dt) for _ in range(4)]
true_positions = [[], [], [], []]
estimated_positions = [[], [], [], []]


# --------------------- KALMAN TRACKING ---------------------
for frame_idx, detected_ranges in enumerate(detected_ranges_each_frame):
    for i, meas in enumerate(detected_ranges):
        true_positions[i].append(meas)
        est = kalman_filters[i].update(meas)
        estimated_positions[i].append(est[0][0])


# --------------------- FUTURE PREDICTION ---------------------
predicted_future_steps = 20
future_predictions = [[] for _ in range(4)]

for i in range(4):
    last_state = kalman_filters[i].x.copy()
    for _ in range(predicted_future_steps):
        last_state = kalman_filters[i].F @ last_state
        future_predictions[i].append(last_state[0][0])


# --------------------- TRACKING PLOT ---------------------
plt.figure(figsize=(10, 5))
colors = ["r", "g", "b", "m"]
labels = ["Car A", "Car B", "Bike", "Pedestrian"]

for i in range(4):
    plt.plot(true_positions[i], marker='o', color=colors[i], label=f"Measured {labels[i]}")
    plt.plot(estimated_positions[i], linestyle='--', color=colors[i], label=f"Kalman Estimated {labels[i]}")
    plt.plot(range(len(estimated_positions[i]), len(estimated_positions[i]) + predicted_future_steps),
             future_predictions[i], linestyle=':', color=colors[i], label=f"Predicted Future {labels[i]}")

plt.title("Multi-Target Kalman Tracking + Future Prediction")
plt.xlabel("Frame")
plt.ylabel("Range (m)")
plt.grid(True)
plt.legend()
plt.show()
