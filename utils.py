import numpy as np

def get_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    angle = np.abs(np.rad2deg(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])))
    if angle > 180:
        angle = 360 - angle
    return angle