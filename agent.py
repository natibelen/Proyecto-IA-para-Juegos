import numpy as np
import win32gui

def get_region(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        raise Exception(f"Window not found: {window_title}")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

    region = {
        "left": left+180,
        "top": top+95,
        "width": right - left - 940,
        "height": bottom - top - 500
    }
    return region


def check_pixel(img, x, y):
    pixel = img[y, x]
    target_color = np.array([205, 77, 10, 255])
    tolerance = 10
    if np.all(np.abs(pixel - target_color) <= tolerance):
        return True







