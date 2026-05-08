import cv2
import numpy as np

def extract_features_from_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = img.shape[0] * img.shape[1]

    yellow_mask = cv2.inRange(img_hsv, (20, 80, 80), (35, 255, 255))
    green_mask  = cv2.inRange(img_hsv, (35, 60, 60), (85, 255, 255))
    brown_mask  = cv2.inRange(img_hsv, (5,  40, 20), (20, 200, 180))
    black_mask  = cv2.inRange(img_hsv, (0,  0,  0),  (180, 255, 50))

    yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
    green_ratio  = cv2.countNonZero(green_mask)  / total_pixels
    brown_ratio  = cv2.countNonZero(brown_mask)  / total_pixels
    black_ratio  = cv2.countNonZero(black_mask)  / total_pixels

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_roughness = float(np.var(laplacian)) / 10000.0

    mean_brightness = float(np.mean(gray))
    std_brightness  = float(np.std(gray))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / total_pixels

    hue_mean        = float(np.mean(img_hsv[:, :, 0]))
    saturation_mean = float(np.mean(img_hsv[:, :, 1]))

    return {
        "yellow_ratio":      yellow_ratio,
        "brown_ratio":       brown_ratio,
        "green_ratio":       green_ratio,
        "black_ratio":       black_ratio,
        "texture_roughness": min(texture_roughness, 1.0),
        "mean_brightness":   mean_brightness,
        "std_brightness":    std_brightness,
        "edge_density":      edge_density,
        "hue_mean":          hue_mean,
        "saturation_mean":   saturation_mean,
    }