import pygame

class AutoPlayer:

    def __init__(self, hit_zone_y: int, perfect_win: int = 10, cooldown_ms: int = 40):
        self.hit_zone_y = hit_zone_y
        self.perfect_win = perfect_win
        self.cooldown_ms = cooldown_ms
        self.last_press_time = {"left": 0, "down": 0, "up": 0, "right": 0}

    def reset(self):
        for k in self.last_press_time:
            self.last_press_time[k] = 0

    def update(self, game):

        now_ms = pygame.time.get_ticks()

        for direction in ("left", "down", "up", "right"):
            candidates = [a for a in game.arrows if a.direction == direction and not a.hit]
            if not candidates:
                continue

            a = min(candidates, key=lambda x: abs(x.y - self.hit_zone_y))
            diff = abs(a.y - self.hit_zone_y)

            if now_ms - self.last_press_time[direction] < self.cooldown_ms:
                continue

            if diff <= self.perfect_win:
                game.handle_input(direction)
                self.last_press_time[direction] = now_ms
