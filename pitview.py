import cv2
import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_closest_object(objects, reference_point):
    distances = []
    for obj in objects:
        distances.append(calculate_distance(obj, reference_point))
    closest_index = np.argmin(distances)
    return objects[closest_index]

def add_text(image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, color, thickness)

# Set the dimensions of the cockpit view
width = 800
height = 600

# Create a blank image with the desired dimensions
cockpit_view = np.zeros((height, width, 3), dtype=np.uint8)

# Draw the cockpit frame
frame_color = (0, 0, 255)  # Red color
frame_thickness = 2
cv2.rectangle(cockpit_view, (50, 50), (width - 50, height - 50), frame_color, frame_thickness)

# Draw the instrument panel
panel_color = (0, 255, 0)  # Green color
panel_thickness = cv2.FILLED
cv2.rectangle(cockpit_view, (100, 100), (width - 100, height - 100), panel_color, panel_thickness)

# Draw some example instruments or indicators
indicator_color = (255, 255, 255)  # White color
indicator_thickness = 1
cv2.circle(cockpit_view, (width // 2, height // 2), 50, indicator_color, indicator_thickness)

cap = cv2.VideoCapture(0)  # Use the appropriate camera index or video file path

while True:
    ret, frame = cap.read()

    # Assuming you have a list of object positions, you can find the closest one
    object_positions = [(100, 200), (300, 400), (500, 600)]  # Example list of object positions
    reference_point = (frame.shape[1] // 2, frame.shape[0] // 2)  # Assuming reference point as the center of the frame

    closest_object = find_closest_object(object_positions, reference_point)

    # Add text on the detected object
    text = "Closest Object"
    add_text(frame, text, closest_object)

    # Display the cockpit view and the frame with text
    cv2.imshow("Cockpit View", cockpit_view)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
