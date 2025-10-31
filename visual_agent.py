import numpy as np
import win32gui
import mss
import mss.tools

class VisualAgent:
    def __init__(self):
        self.latest_frame = None
        self.stop_capture = False
        self.wait_time = 0

    def check_visual_agent(self, game):
        if self.latest_frame is not None and self.wait_time == 0:
            if check_pixel(self.latest_frame, 62, 40):
                game.handle_input("left")
            elif check_pixel(self.latest_frame, 128, 4):
                game.handle_input("down")
            elif check_pixel(self.latest_frame, 228, 73):
                game.handle_input("up")
            elif check_pixel(self.latest_frame, 296, 40):
                game.handle_input("right")
            self.wait_time = 1
        elif self.wait_time > 0:
            self.wait_time -=1

    def take_screenshot(self):
        sct = mss.mss()
        while not self.stop_capture:
            region = get_region("Dance Dance ReMixed")
            img = np.array(sct.grab(region))
            self.latest_frame = img

    def get_frame(self):
        return self.latest_frame

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









