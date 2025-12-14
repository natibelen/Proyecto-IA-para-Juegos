import csv
import time

ACTION_TO_ID = {"none": 0, "left": 1, "down": 2, "up": 3, "right": 4}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}

class DemoRecorder:
    def __init__(self, path="demos.csv"):
        self.path = path
        self.rows = []
        self.enabled = False
        self.start_time = None

    def start(self):
        self.rows = []
        self.enabled = True
        self.start_time = time.time()
        print(f"[REC] Recording ON -> {self.path}")

    def stop_and_save(self):
        if not self.enabled:
            return
        self.enabled = False

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["t", "dy_left", "dy_down", "dy_up", "dy_right", "action"])
            w.writerows(self.rows)

        print(f"[REC] Saved {len(self.rows)} frames to {self.path}")
        self.rows = []

    def add(self, t, state_vec, action_id):
        if not self.enabled:
            return
        self.rows.append([t, *state_vec, action_id])
