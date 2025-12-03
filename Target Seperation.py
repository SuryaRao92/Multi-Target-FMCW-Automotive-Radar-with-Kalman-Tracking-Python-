dt = T_chirp  # time step between frames

kalman_filters = [KalmanFilter(dt) for _ in range(4)]
true_positions = []
estimated_positions = []
predicted_positions = []

for frame in range(Nc):

    # Suppose detected_ranges array holds peak results from FFT
    # Example: detected_ranges = [80, 120, 60, 40]

    for i, meas in enumerate(detected_ranges):
        true_positions.append(meas)

        pred = kalman_filters[i].predict()
        est = kalman_filters[i].update(meas)

        predicted_positions.append(pred[0][0])
        estimated_positions.append(est[0][0])
