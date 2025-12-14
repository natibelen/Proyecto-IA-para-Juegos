import torch
import torch.nn as nn
import numpy as np

ID_TO_ACTION = {0: "none", 1: "left", 2: "down", 3: "up", 4: "right"}

class MLP(nn.Module):
    def __init__(self, in_dim=4, hidden=64, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class ImitationAgent:
    def __init__(self, model_path="bc_model.pt", cooldown_ms=40, min_conf=0.55):
        self.model = MLP()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.cooldown_ms = cooldown_ms
        self.min_conf = min_conf
        self.last_press_ms = {"left": 0, "down": 0, "up": 0, "right": 0}

    def reset(self):
        for k in self.last_press_ms:
            self.last_press_ms[k] = 0

    @torch.no_grad()
    def act(self, state_vec):
        x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # (1,4)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        action_id = int(np.argmax(probs))
        conf = float(probs[action_id])
        return action_id, conf

    def update(self, game, now_ms, state_vec):
        action_id, conf = self.act(state_vec)

        if action_id == 0:
            return

        if conf < self.min_conf:
            return

        direction = ID_TO_ACTION[action_id]
        if now_ms - self.last_press_ms[direction] < self.cooldown_ms:
            return

        game.handle_input(direction)
        self.last_press_ms[direction] = now_ms
