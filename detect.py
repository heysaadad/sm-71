import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
radar_display = False
prev_centroid = None
locked_target = None
window_name = "SM-71 Drone Radar"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)
scan_line_position = 0
scan_line_direction = 1
target_indicator_blink = False
target_indicator_blink_duration = 0.0005  # in seconds
target_indicator_last_blink_time = time.time()

missile_name = "SM-71 Drone Radar"

while True:
    ret, frame = cap.read()
    fg_mask = bg_subtractor.apply(frame)
    _, binary_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        radar_display = True
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
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                current_centroid = np.array((cx, cy))
                if prev_centroid is None:
                    prev_centroid = current_centroid
                else:
                    current_centroid = (prev_centroid + current_centroid) // 2
                    prev_centroid = current_centroid
                if locked_target is None:
                    locked_target = current_centroid
                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Add object title
                object_title = f"Object {len(contours)}"
                cv2.putText(frame, object_title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                current_time = time.time()
                if current_time - target_indicator_last_blink_time >= target_indicator_blink_duration:
                    target_indicator_blink = not target_indicator_blink
                    target_indicator_last_blink_time = current_time

                if target_indicator_blink:
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                                   markerSize=30, thickness=2)

    else:
        radar_display = False
        locked_target = None
    radar_overlay = np.zeros_like(frame)

    if radar_display:
        radar_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        radar_radius = min(frame.shape[0] // 2, frame.shape[1] // 2) - 50
        cv2.circle(radar_overlay, radar_center, radar_radius, (0, 255, 0), 2)
        cv2.circle(radar_overlay, radar_center, int(radar_radius * 0.75), (0, 255, 0), 1)
        cv2.circle(radar_overlay, radar_center, int(radar_radius * 0.5), (0, 255, 0), 1)
        beam_length = radar_radius + 50
        for angle in range(0, 360, 10):
            start_point = (int(radar_center[0] + radar_radius * np.cos(np.radians(angle))),
                           int(radar_center[1] + radar_radius * np.sin(np.radians(angle))))
            end_point = (int(radar_center[0] + beam_length * np.cos(np.radians(angle))),
                         int(radar_center[1] + beam_length * np.sin(np.radians(angle))))
            cv2.line(radar_overlay, start_point, end_point, (0, 255, 0), 1)

    frame = cv2.addWeighted(frame, 0.7, radar_overlay, 0.3, 0)
    scan_line_color = (0, 255, 0)
    cv2.line(frame, (0, scan_line_position), (frame.shape[1], scan_line_position), scan_line_color, 1)
    scan_line_position += scan_line_direction
    if scan_line_position <= 0 or scan_line_position >= frame.shape[0]:
        scan_line_direction *= -1
    cv2.putText(frame, missile_name, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
