import numpy as np
import cv2


class ImageProcessing:
    @staticmethod
    def draw_line(img, theta, rho):
        h, w = img.shape[:2]
        if np.isclose(np.sin(theta), 0):
            x1, y1 = rho, 0
            x2, y2 = rho, h
        else:
            def calc_y(x): return rho / np.sin(theta) - \
                x * np.cos(theta) / np.sin(theta)
            x1, y1 = 0, calc_y(0)
            x2, y2 = w, calc_y(w)

        # float -> int
        x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
