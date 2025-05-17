from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time
import serial

def equalize_y_channel(frame):
    frame_yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
    y, u, v = cv.split(frame_yuv)
    y_eq = clahe.apply(y)
    frame_yuv_eq = cv.merge((y_eq, u, v))
    frame_output = cv.cvtColor(frame_yuv_eq, cv.COLOR_YUV2BGR)
    return frame_output

# Initialize serial connection
arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)


# CLAHE setup
clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(5, 5))

# Picamera2 setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640*2, 480*2)}))
picam2.start()  # Let camera warm up

# FPS tracking
prev_time = time.time()
frame_count = 0
fps = 0

# HSV color bounds for green
lower_green = np.array([35, 40, 0])
upper_green = np.array([85, 255, 255])
kernel = np.ones((5, 5), np.uint8)

# PID parameters for horizontal offset
Kp_x = 0.2
Ki_x = 0.05
Kd_x = 0
integral_x = 0
prev_error_x = 0

# PID parameters for distance (area)
Kp_a = 0.01
Ki_a = 0.01
Kd_a = 0.02
integral_a = 0
prev_error_a = 0

# Control thresholds
offset_threshold = 50
max_pwm = 200
min_area = 300
max_area = 5000

# Command throttling
command_timer = 0
command_interval = 5

while True:
    frame = picam2.capture_array()
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0

    frame = cv.flip(frame, 1)
    scaled1 = cv.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
    scaled = equalize_y_channel(scaled1)

    hsv = cv.cvtColor(scaled, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_green, upper_green)
    segmented = cv.bitwise_and(scaled, scaled, mask=mask)

    gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
    _, binary_mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)
    binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

    command = b"S000\n"  # Default command to stop

    if valid_contours:
        largest = max(valid_contours, key=cv.contourArea)
        area = cv.contourArea(largest)
        cv.drawContours(segmented, [largest], -1, (0, 255, 0), 3)

        M = cv.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(segmented, (cx, cy), 5, (0, 0, 255), -1)

            frame_center_x = scaled.shape[1] // 2
            horizontal_offset = cx - frame_center_x
            cv.putText(segmented, f"Offset: {horizontal_offset}px", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv.line(segmented, (frame_center_x, 0), (frame_center_x, scaled.shape[0]), (255, 0, 0), 2)

            # PID Control for horizontal alignment
            error_x = horizontal_offset
            integral_x += error_x
            derivative_x = error_x - prev_error_x
            output_x = Kp_x * error_x + Ki_x * integral_x + Kd_x * derivative_x
            prev_error_x = error_x

            if abs(error_x) > offset_threshold:
                pwm = min(max(abs(int(output_x)), 0), max_pwm)
                direction = 'L' if error_x > 0 else 'R'
                command = f"{direction}{pwm:03}\n".encode()
            else:
                # PID for forward motion (distance)
                error_a = max_area - area
                integral_a += error_a
                derivative_a = error_a - prev_error_a
               #output_a = Kp_a * error_a + Ki_a * integral_a + Kd_a * derivative_a
                output_a=100
                prev_error_a = error_a

                if min_area < area < max_area:
                    pwm = min(max(int(output_a), 0), max_pwm)
                    direction = 'F' if area < max_area else 'S'
                    command = f"{direction}{pwm:03}\n".encode()
                else:
                    command = b"S000\n"
        else:
            command = b"S000\n"
    else:
        command = b"S000\n"

    # === Send command only every N frames ===
    if command_timer % command_interval == 0:
        arduino.write(command)
        print(f"Sent: {command.decode().strip()}")

    command_timer += 1

    # Display outputs
    cv.putText(segmented, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow('Segmented', segmented)
    cv.imshow('Mask', binary_mask)
    cv.imshow('Original', scaled)
    cv.imshow('og', scaled1)

    if cv.waitKey(1) & 0xFF == ord('q'):
        command = b"S000\n"
        arduino.write(command)
        break

picam2.stop()
arduino.close()
cv.destroyAllWindows()

