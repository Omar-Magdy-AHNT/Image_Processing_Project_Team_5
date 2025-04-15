from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time
import serial

# Initialize serial connection to Arduino
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Update for Linux device path
time.sleep(2)

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)  # Allow camera to warm up

# FPS setup
prev_time = time.time()
frame_count = 0
fps = 0

# Color bounds for green object (in HSV)
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Morphological kernel
kernel = np.ones((5, 5), np.uint8)

# Control parameters
offset_threshold = 30  # Minimum offset before action
Kp = 1               # Proportional gain
max_pwm = 255          # Max PWM value

while True:
    frame = picam2.capture_array()
    frame = cv.flip(frame, 1)
    
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - prev_time

    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0

    scaled = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

    hsv = cv.cvtColor(scaled, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_green, upper_green)
    segmented = cv.bitwise_and(scaled, scaled, mask=mask)

    gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
    _, binary_mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv.contourArea)
        cv.drawContours(segmented, [largest], -1, (0, 255, 0), 3)

        M = cv.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(segmented, (cx, cy), 5, (0, 0, 255), -1)

            frame_center_x = scaled.shape[1] // 2
            horizontal_offset = cx - frame_center_x
            cv.putText(segmented, f"Offset: {horizontal_offset}px", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
            cv.line(segmented, (frame_center_x, 0), (frame_center_x, scaled.shape[0]), (255, 0, 0), 2)

            # === P-Control Logic ===
            if abs(horizontal_offset) > offset_threshold:
                pwm = min(int(Kp * abs(horizontal_offset)), max_pwm)
                direction = 'L' if horizontal_offset > 0 else 'R'
                command = f"{direction}{pwm:03}\n".encode()
                arduino.write(command)
            else:
                arduino.write(b"S000\n")

    cv.putText(segmented, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow('Segmented', segmented)

    if cv.waitKey(1) & 0xFF == ord('q'):
        arduino.write(b"S000\n")
        break

arduino.close()
cv.destroyAllWindows()
