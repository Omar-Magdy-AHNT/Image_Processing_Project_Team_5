import cv2
import time

# New GStreamer pipeline for better compatibility
#gst_pipeline = (
 #   "libcamerasrc ! video/x-raw, width=1920, height=1080, framerate=30/1 ! "
 #   "videoconvert ! videoscale ! video/x-raw, width=320, height=240 ! appsink"
#)
gst_pipeline = (
    "libcamerasrc ! videoconvert ! appsink"
)







# Open the camera stream
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        small_frame = cv2.resize(frame, (320, 240))  # Change size here
        # Show the live camera feed
        cv2.imshow("Live Camera Feed", small_frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Press 's' to capture a photo
            filename = f"photo_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Photo saved as {filename}")

        elif key == ord('q'):  # Press 'q' to quit
            break

finally:
    # Release resources properly
    cap.release()
    cv2.destroyAllWindows()
