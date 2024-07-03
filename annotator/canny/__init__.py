import cv2


class CannyDetector:
    def __call__(self, img, low_threshold=100, high_threshold=200):
        return cv2.Canny(img, low_threshold, high_threshold)
