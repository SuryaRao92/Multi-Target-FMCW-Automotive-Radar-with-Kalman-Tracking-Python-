import numpy as np

class KalmanFilter:
    def __init__(self, dt, process_noise=1, measurement_noise=5):
        self.dt = dt

        # State vector [range, velocity]
        self.x = np.array([[0], [0]])

        # State transition model
        self.F = np.array([[1, dt],
                           [0, 1]])

        # Measurement model: we measure only range
        self.H = np.array([[1, 0]])

        # Covariance matrices
        self.P = np.eye(2)
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])

    # Predict phase
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    # Update phase
    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x
