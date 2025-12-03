# 🚗 Multi-Target FMCW Automotive Radar with Kalman Tracking (Python)

### 📌 Project Overview
This project simulates an FMCW (Frequency Modulated Continuous Wave) automotive radar system capable of detecting and tracking **multiple moving objects** using **Range–Doppler processing** and **Kalman filtering**.

### 🎯 Objectives
- Simulate FMCW radar signals for multiple moving targets
- Detect target **range and velocity** using 1D & 2D FFT
- Implement **Kalman Filter** to track objects over time
- Predict future motion for **collision avoidance / ADAS safety**

### 🛰 Real-World Use Cases
- Self-driving cars (ADAS)
- Pedestrian detection
- Drone navigation
- Military surveillance radar

---

## 🧠 Features
- Multi-target motion simulation (Car A, Car B, Bike, Pedestrian)
- Range FFT detection
- Range–Doppler heatmap visualization
- Kalman filtering for tracking & smoothing
- Future trajectory prediction

---

## 🧮 Tech Stack
| Category | Tools |
|----------|--------|
| Programming | Python |
| Libraries | NumPy, Matplotlib |
| DSP Concepts | FFT, Doppler, Radar chirps |
| Estimation | Kalman Filter |

---

## 📊 Output Plots
- Range Profile & Range-Doppler Map
- Multi-target tracking graph
- Future path prediction visualization

---

## 🚀 How to Run
```bash
pip install numpy matplotlib
python fmcw_radar_tracking.py

