import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
altitude_meter_display = False
prev_altitude = None
window_name = "SM-71 Drone Altitude Meter"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)
scan_line_position = 0
scan_line_direction = 1
altitude_indicator_blink = False
altitude_indicator_blink_duration = 0.0005  # in seconds
altitude_indicator_last_blink_time = time.time()

drone_name = "SM-71 Drone"

altitude = 0
gyro_angle = 0

while True:
    ret, frame = cap.read()
    fg_mask = bg_subtractor.apply(frame)
    _, binary_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        altitude_meter_display = True
        best_contour = None
        best_contour_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            if area > best_contour_area:
                best_contour = contour
                best_contour_area = area

        if best_contour is not None:
            moments = cv2.moments(best_contour)
            if moments["m00"] != 0:
                cy = int(moments["m01"] / moments["m00"])
                current_altitude = frame.shape[0] - cy
                if prev_altitude is None:
                    prev_altitude = current_altitude
                else:
                    current_altitude = (prev_altitude + current_altitude) // 2
                    prev_altitude = current_altitude
                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Add object title
                object_title = f"Object {len(contours)}"
                cv2.putText(frame, object_title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                current_time = time.time()
                if current_time - altitude_indicator_last_blink_time >= altitude_indicator_blink_duration:
                    altitude_indicator_blink = not altitude_indicator_blink
                    altitude_indicator_last_blink_time = current_time

                if altitude_indicator_blink:
                    cv2.drawMarker(frame, (x + w // 2, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                                   markerSize=30, thickness=2)

    else:
        altitude_meter_display = False

    # Add altitude meter
    meter_overlay = np.zeros_like(frame)

    if altitude_meter_display:
        meter_start = frame.shape[0] - 50
        meter_end = frame.shape[0] - 150
        cv2.line(meter_overlay, (100, meter_start), (100, meter_end), (0, 255, 0), 2)
        cv2.line(meter_overlay, (90, meter_start), (110, meter_start), (0, 255, 0), 2)
        cv2.line(meter_overlay, (90, meter_end), (110, meter_end), (0, 255, 0), 2)
        cv2.putText(meter_overlay, "0", (60, meter_start + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(meter_overlay, "100", (40, meter_end - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(meter_overlay, "Altitude (m)", (10, meter_start - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.addWeighted(frame, 0.7, meter_overlay, 0.3, 0)

    # Add mini radar
    mini_radar_overlay = np.zeros((150, 150, 3), dtype=np.uint8)

    if altitude_meter_display:
        mini_radar_center = (mini_radar_overlay.shape[1] // 2, mini_radar_overlay.shape[0] // 2)
        mini_radar_radius = min(mini_radar_overlay.shape[0] // 2, mini_radar_overlay.shape[1] // 2) - 10
        cv2.circle(mini_radar_overlay, mini_radar_center, mini_radar_radius, (0, 255, 0), 2)
        cv2.circle(mini_radar_overlay, mini_radar_center, int(mini_radar_radius * 0.75), (0, 255, 0), 1)
        cv2.circle(mini_radar_overlay, mini_radar_center, int(mini_radar_radius * 0.5), (0, 255, 0), 1)
        beam_length = mini_radar_radius + 10
        for angle in range(0, 360, 10):
            start_point = (int(mini_radar_center[0] + mini_radar_radius * np.cos(np.radians(angle))),
                           int(mini_radar_center[1] + mini_radar_radius * np.sin(np.radians(angle))))
            end_point = (int(mini_radar_center[0] + beam_length * np.cos(np.radians(angle))),
                         int(mini_radar_center[1] + beam_length * np.sin(np.radians(angle))))
            cv2.line(mini_radar_overlay, start_point, end_point, (0, 255, 0), 1)

    frame[:150, -150:, :] = mini_radar_overlay

    # Add angle and rotation meter
    angle_meter_overlay = np.zeros((150, 150, 3), dtype=np.uint8)

    if altitude_meter_display:
        angle_center = (angle_meter_overlay.shape[1] // 2, angle_meter_overlay.shape[0] // 2)
        angle_radius = min(angle_meter_overlay.shape[0] // 2, angle_meter_overlay.shape[1] // 2) - 10
        cv2.circle(angle_meter_overlay, angle_center, angle_radius, (0, 255, 0), 2)
        angle_start_point = (int(angle_center[0] + angle_radius * np.cos(np.radians(-gyro_angle))),
                             int(angle_center[1] + angle_radius * np.sin(np.radians(-gyro_angle))))
        cv2.line(angle_meter_overlay, angle_center, angle_start_point, (0, 255, 0), 2)
        cv2.putText(angle_meter_overlay, "0", (angle_center[0] - 20, angle_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)
        cv2.putText(angle_meter_overlay, "180", (angle_center[0] + angle_radius - 30, angle_center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(angle_meter_overlay, "Angle", (angle_center[0] - 25, angle_center[1] - angle_radius - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    frame[frame.shape[0] // 2 - 75:frame.shape[0] // 2 + 75, frame.shape[1] // 2 - 75:frame.shape[1] // 2 + 75, :] = angle_meter_overlay

    scan_line_color = (0, 255, 0)
    cv2.line(frame, (0, scan_line_position), (frame.shape[1], scan_line_position), scan_line_color, 1)
    scan_line_position += scan_line_direction
    if scan_line_position <= 0 or scan_line_position >= frame.shape[0]:
        scan_line_direction *= -1

    # Add altitude and angle information
    altitude_text = f"Altitude: {altitude} m"
    cv2.putText(frame, altitude_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    gyro_text = f"Angle: {gyro_angle} degrees"
    cv2.putText(frame, gyro_text, (frame.shape[1] // 2 - 100, frame.shape[0] // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
