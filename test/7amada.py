import cv2
import numpy as np
import time

# Toggle to show internal processing masks (set True for debugging)
SHOW_DEBUG = False


def preprocess(frame, clahe, kernel_open, kernel_close):
    """Apply CLAHE + adaptive threshold + morphology to obtain a binary mask.

    Steps:
      1. Convert to grayscale (keeps full frame for adaptive threshold stability)
      2. CLAHE for local contrast
      3. Gaussian blur (denoise)
      4. Adaptive threshold (binary) -> morphological open/close to clean
    Returns the cleaned binary image (uint8 0/255).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    return closed


def green_mask(frame):
    """Optional helper: produce a green mask in HSV to reinforce detection.
    This is ANDed later with the adaptive mask to suppress non-green blobs.
    Adjust ranges as needed for your lighting.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Two ranges can be used if green spans wide hue; start with a core range.
    lower_green = np.array([35, 60, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask


def select_ball_via_ccl(binary_mask, color_frame=None, offset=(0, 0)):
    """Use connected component labeling to select the most likely circular green ball.

    Heuristics applied per component:
      - Area threshold (min_area)
      - Aspect ratio near 1 (max aspect deviation)
      - Fill ratio (area / (w*h)) close to a circle (~0.785). We accept a range.
      - (Optional) color consistency if color_frame provided: mean green channel > R & B.

    offset: (ox, oy) added to centroid & bbox to map ROI coords back to full frame.

    Returns: (cx, cy, bbox, mask_component) or (None, None, None, None)
    """
    min_area = 200
    fill_min, fill_max = 0.5, 0.95
    max_aspect_dev = 0.35  # allows some perspective distortion

    # connectedComponentsWithStats expects 0/255; ensure binary
    _, bw = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    best = None
    best_score = -1

    for label in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        aspect = w / float(h) if h > 0 else 999
        aspect_dev = abs(aspect - 1)
        if aspect_dev > max_aspect_dev:
            continue
        fill_ratio = area / float(w * h)
        if not (fill_min <= fill_ratio <= fill_max):
            continue

        score = area * (1 - aspect_dev) * fill_ratio  # simple combined metric

        # Optional color check for further discrimination
        if color_frame is not None:
            comp_mask = (labels == label).astype(np.uint8)
            # Mean channel values inside component
            b_mean = cv2.mean(color_frame[:, :, 0], mask=comp_mask)[0]
            g_mean = cv2.mean(color_frame[:, :, 1], mask=comp_mask)[0]
            r_mean = cv2.mean(color_frame[:, :, 2], mask=comp_mask)[0]
            if not (g_mean > r_mean * 1.05 and g_mean > b_mean * 1.05):
                continue
            score *= g_mean  # boost greener blobs

        if score > best_score:
            cx, cy = centroids[label]
            ox, oy = offset
            best = (
                int(cx + ox),
                int(cy + oy),
                (x + ox, y + oy, w, h),
                (labels == label).astype(np.uint8) * 255,
            )
            best_score = score

    if best is None:
        return None, None, None, None
    return best


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # For smoothing centroid (simple exponential moving average)
    alpha = 0.25
    smooth_cx, smooth_cy = None, None
    frame_count = 0
    last_time = time.time()
    fps = None  # smoothed FPS

    # ROI parameters
    roi_half_size_initial = 120  # starting half-size for search window
    roi_half_size_max = 240
    roi_half_size_min = 60
    roi_expand_factor = 1.3
    roi_shrink_factor = 0.85
    lost_frames = 0
    max_lost_before_full_reset = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # Decide whether to use full-frame or ROI processing
        use_roi = smooth_cx is not None and smooth_cy is not None and lost_frames < max_lost_before_full_reset
        roi_half = roi_half_size_initial
        if use_roi:
            # Expand ROI with number of lost frames
            roi_half = int(
                min(
                    roi_half_size_max,
                    max(
                        roi_half_size_min,
                        roi_half_size_initial * (roi_expand_factor ** lost_frames),
                    ),
                )
            )
            cx_guess, cy_guess = smooth_cx, smooth_cy
            x0 = max(0, cx_guess - roi_half)
            y0 = max(0, cy_guess - roi_half)
            x1 = min(W, cx_guess + roi_half)
            y1 = min(H, cy_guess + roi_half)
            roi_frame = frame[y0:y1, x0:x1]
            binary_mask_roi = preprocess(roi_frame, clahe, kernel_open, kernel_close)
            gmask_roi = green_mask(roi_frame)
            combined_roi = cv2.bitwise_and(binary_mask_roi, gmask_roi)
            cx, cy, bbox, comp_mask = select_ball_via_ccl(
                combined_roi, color_frame=roi_frame, offset=(x0, y0)
            )
            if cx is None:
                # Fallback to full frame if ROI failed
                binary_mask = preprocess(frame, clahe, kernel_open, kernel_close)
                gmask = green_mask(frame)
                combined = cv2.bitwise_and(binary_mask, gmask)
                cx, cy, bbox, comp_mask = select_ball_via_ccl(combined, color_frame=frame)
            else:
                binary_mask = binary_mask_roi  # for display (cropped)
                gmask = gmask_roi
                combined = combined_roi
        else:
            binary_mask = preprocess(frame, clahe, kernel_open, kernel_close)
            gmask = green_mask(frame)
            combined = cv2.bitwise_and(binary_mask, gmask)
            cx, cy, bbox, comp_mask = select_ball_via_ccl(combined, color_frame=frame)

        # --- FPS measurement (start) ---
        now = time.time()
        dt = now - last_time
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = inst_fps if fps is None else (0.9 * fps + 0.1 * inst_fps)
        last_time = now

        if cx is not None:
            if smooth_cx is None:
                smooth_cx, smooth_cy = cx, cy
            else:
                smooth_cx = int(alpha * cx + (1 - alpha) * smooth_cx)
                smooth_cy = int(alpha * cy + (1 - alpha) * smooth_cy)

            x, y, w, h = bbox
            radius = int(0.25 * (w + h))  # approximate radius
            cv2.circle(frame, (smooth_cx, smooth_cy), radius, (0, 255, 0), 2)
            cv2.circle(frame, (smooth_cx, smooth_cy), 5, (0, 0, 255), -1)

            lost_frames = 0  # reset lost counter

            if frame_count % 10 == 0:  # throttle console output
                print(
                    f"Ball Centroid (smoothed): ({smooth_cx}, {smooth_cy}) | ROI half: {roi_half} | lost: {lost_frames}"
                )

            # Show extracted ball crop
            crop = frame[y : y + h, x : x + w]
            if crop.size > 0:
                cv2.imshow("Extracted Ball", crop)
        else:
            lost_frames += 1
            if frame_count % 10 == 0:
                print(f"Tracking lost ({lost_frames}) - expanding ROI / may reset soon")
            if lost_frames >= max_lost_before_full_reset:
                smooth_cx, smooth_cy = None, None  # force full frame search next loop

        # Visualization windows
        if SHOW_DEBUG:
            cv2.imshow("Binary (Adaptive + Morph)", binary_mask)
            cv2.imshow("Green Mask", gmask)
            cv2.imshow("Combined Mask", combined)
        # Overlay FPS (after drawing tracking visuals)
        if fps is not None:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 255, 50),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Frame", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""Your current pipeline (simplified):
1. Frame → Grayscale + CLAHE + Blur + Adaptive threshold.
2. HSV green mask.
3. Bitwise AND (adaptive mask & green mask).
4. Connected components + simple shape heuristics (area, aspect, fill).
5. Exponential moving average smoothing (EMA) + expanding ROI when lost.

Typical star‑tracker (pre–attitude) pipeline:
1. Sensor calibration: dark-frame subtraction, flat-field, hot pixel map, lens undistort.
2. Background / illumination estimation (large Gaussian or median) → residual image.
3. Noise/statistics: estimate σ (robust/MAD) of residual; threshold = μ + k·σ (k≈4–6) to get candidate stars.
4. Blob refinement: remove hot pixels, morphology, connected components; quality metrics (SNR, roundness, FWHM, saturation).
5. Sub-pixel centroid (intensity-weighted or Gaussian fit) + quality gating; (then later identification/attitude—not in scope).

What you have vs star-tracker:
- Calibration: Missing (you start raw; star trackers correct sensor first).
- Thresholding: Adaptive local threshold (heuristic) vs statistically driven k·σ method.
- Color: You depend on HSV color; star trackers are monochrome, color-agnostic (use brightness + noise model).
- Intersection (AND): You combine two masks to suppress non-green; star trackers rely on SNR/shape metrics instead of color logic.
- Component scoring: You use area * fill * aspect; star trackers weight SNR, roundness, PSF width, saturation.
- Centroid: You use bounding box average radius + integer center after smoothing; star trackers do sub-pixel intensity-weighted (higher precision).
- Temporal smoothing: EMA (no motion model) vs predictive filter (Kalman/EKF) with residual-based noise tuning.
- Recovery: ROI expansion only vs state machine (ACQUIRE → TRACK → LOST) + adaptive thresholds and hot-pixel rejection.
- Metrics/logging: Minimal (print every 10 frames) vs detailed logs (σ, SNR, centroid error, flags).

What is “right” (acceptable) in your current steps:
- Using ROI to reduce search space (concept similar to limiting processing window).
- Morphology & connected components (standard).
- Basic shape filtering (aspect/fill) is a lightweight proxy for roundness.

What is weak relative to star-tracker standards:
- Adaptive threshold (can over/under-segment; not noise-model-based).
- Reliance on color (brittle under lighting shifts; not transferable).
- No sensor calibration (systematic errors unaddressed).
- Integer / coarse centroid and approximate radius → lower positional precision.
- No SNR / noise metrics → no principled confidence.
- EMA smoothing only (lag, no predictive capability, no outlier gating).
- No quality or health flags (cannot claim reliability).

What to adopt from star-trackers (practical for your case):
1. Dark + (optional) flat correction (even a single dark frame improves consistency).
2. Background removal + residual noise σ estimation (replace adaptive threshold).
3. Statistical threshold (resid > k·σ) plus min area filter (color optional secondary gate).
4. Sub-pixel centroid (intensity-weighted) for smoother control.
5. Roundness (circularity) and SNR metrics → confidence score.
6. Kalman filter (x,y,vx,vy) with residual-based R tuning and outlier rejection.
7. State machine (ACQUIRE/TRACK/LOST) with adaptive search expansion rules.
8. Logging of per-frame metrics (σ, SNR, circularity, confidence, state) to justify “process quality.”

Optional “extra credibility” items:
- Hot pixel suppression (remove single-pixel bright spikes).
- Saturation flag (detect if highlight clipping occurs in the blob).
- Latency-aware prediction (predict to actuation time).

Phrase you can honestly use:
“I’m applying a star-tracker inspired preprocessing chain: sensor calibration (dark/flat), background & noise modeling, statistical (k·σ) detection, sub-pixel centroiding, quality metrics (SNR, circularity) and a predictive Kalman tracking state machine—adapted to a colored spherical target instead of stars.”

Summary gap list (fix these to be “star-tracker style”):
- Replace adaptive threshold with noise-model threshold.
- Add calibration (dark/flat).
- Implement sub-pixel centroid + SNR & circularity.
- Introduce Kalman + state machine + logging.
- Treat HSV as auxiliary, not core.

Ask if you want a compact checklist to implement in order or a one-line “claim” statement refined further."""