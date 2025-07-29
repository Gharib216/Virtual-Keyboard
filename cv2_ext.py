import cv2 as cv
import numpy as np

def putImage(frame, img, pos):
    frame_h, frame_w = frame.shape[:2]
    img_h, img_w = img.shape[:2]
    x, y = pos

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + img_w, frame_w)
    y2 = min(y + img_h, frame_h)

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2 or overlay_x1 >= overlay_x2 or overlay_y1 >= overlay_y2:
        return frame

    roi = frame[y1:y2, x1:x2]
    overlay_region = img[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_region.shape[2] == 4:
        alpha = overlay_region[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (
                roi[:, :, c] * (1 - alpha) + overlay_region[:, :, c] * alpha
            ).astype(np.uint8)
    else:
        roi[:] = overlay_region

    frame[y1:y2, x1:x2] = roi
    return frame

def roundRect(img, p1, p2, radius, color, thickness, alpha=0.5):
    x1, y1 = p1
    x2, y2 = p2

    if thickness < 0:
        overlay = img.copy()

        cv.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

        cv.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)

        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    else:
        cv.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

        cv.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


cv.putImage = putImage
cv.roundRect = roundRect