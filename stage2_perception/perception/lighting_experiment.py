import cv2
import numpy as np


def apply_gamma(image, gamma=1.8):
    gamma_table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, gamma_table)


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def simple_dehaze(image, haze_strength=0.15):
    img = image.astype(np.float32) / 255.0
    dark_channel = cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
    kernel = np.ones((15, 15), dtype=np.uint8)
    dark_channel = cv2.erode(dark_channel, kernel)
    dark_channel = np.clip(dark_channel, 1e-6, 1.0)

    transmission = 1.0 - haze_strength * dark_channel
    transmission = np.clip(transmission, 0.2, 1.0)

    airlight = np.percentile(img.reshape(-1, 3), 99, axis=0)
    restored = np.empty_like(img)
    for ch in range(3):
        restored[:, :, ch] = (img[:, :, ch] - airlight[ch]) / transmission + airlight[ch]

    restored = np.clip(restored, 0.0, 1.0)
    return (restored * 255).astype(np.uint8)


def add_synthetic_fog(image, fog_intensity=0.25):
    fog_base = np.full_like(image, 220, dtype=np.uint8)
    fog_layer = cv2.GaussianBlur(fog_base, (31, 31), 0)
    alpha = np.clip(fog_intensity, 0.0, 0.8)
    result = cv2.addWeighted(image, 1.0 - alpha, fog_layer, alpha, 0)
    return result


def preprocess_scene(image, fog_level=0.0):
    processed = apply_bilateral_filter(image, d=9, sigma_color=80, sigma_space=80)
    processed = apply_clahe(processed)
    processed = simple_dehaze(processed, haze_strength=max(0.08, 0.35 * fog_level))
    return processed


def detect_blobs(image, label):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 40], dtype=np.uint8)
    upper = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\n--- {label} ---")
    print(f"Found {len(contours)} contours above threshold")

    strong_contours = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 200:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                strong_contours.append((i, cx, cy, area))
                print(f"Contour {i}: centroid=({cx:.1f}, {cy:.1f}), area={area:.1f}")

    if not strong_contours:
        print("No strong contours detected. Visibility is degraded by haze, darkness, or shadowing.")

    return len(strong_contours)


def main():
    img = cv2.imread("aruco_marker_0.png")
    if img is None:
        raise FileNotFoundError("Need aruco_marker_0.png in the project folder")

    bright = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    dark = cv2.convertScaleAbs(img, alpha=0.6, beta=-35)
    dark_gamma = apply_gamma(dark, gamma=2.2)
    low_light_enhanced = apply_clahe(dark_gamma)

    fog_10 = add_synthetic_fog(img, fog_intensity=0.10)
    fog_25 = add_synthetic_fog(img, fog_intensity=0.25)
    fog_40 = add_synthetic_fog(img, fog_intensity=0.40)

    fog_10_processed = preprocess_scene(fog_10, fog_level=0.10)
    fog_25_processed = preprocess_scene(fog_25, fog_level=0.25)
    fog_40_processed = preprocess_scene(fog_40, fog_level=0.40)

    cv2.imwrite("bright_version.png", bright)
    cv2.imwrite("dark_version.png", dark)
    cv2.imwrite("dark_gamma_version.png", dark_gamma)
    cv2.imwrite("low_light_enhanced.png", low_light_enhanced)
    cv2.imwrite("fog_10.png", fog_10)
    cv2.imwrite("fog_25.png", fog_25)
    cv2.imwrite("fog_40.png", fog_40)
    cv2.imwrite("fog_10_processed.png", fog_10_processed)
    cv2.imwrite("fog_25_processed.png", fog_25_processed)
    cv2.imwrite("fog_40_processed.png", fog_40_processed)

    print("Scene perception robustness baseline")
    print("=================================")

    detect_blobs(img, "normal")
    detect_blobs(bright, "bright")
    detect_blobs(dark, "dark")
    detect_blobs(dark_gamma, "dark with gamma correction")
    detect_blobs(low_light_enhanced, "low-light enhanced")
    detect_blobs(fog_10_processed, "fog overlay 10% + preprocessing")
    detect_blobs(fog_25_processed, "fog overlay 25% + preprocessing")
    detect_blobs(fog_40_processed, "fog overlay 40% + preprocessing")


if __name__ == "__main__":
    main()