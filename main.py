import pygame
import random
import time
import cv2
from collections import defaultdict
import os
import pickle

# === Q-learning Agente DDR ====================================================
ACTIONS = ["left", "down", "up", "right", "none"]

class DDRAgent:
    def __init__(self, alpha=0.15, gamma=0.9, epsilon=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)
        self.total_episodes = 0
        self.training_scores = []

    def _key(self, state, action):
        return (tuple(state), action)

    def choose_action(self, state):
        import random
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        best_a, best_q = "none", -1e9
        for a in ACTIONS:
            q = self.Q[self._key(state, a)]
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def update_q_value(self, state, action, reward, next_state):
        max_next = max(self.Q[self._key(next_state, a)] for a in ACTIONS)
        key = self._key(state, action)
        self.Q[key] += self.alpha * (reward + self.gamma * max_next - self.Q[key])

    def get_reward(self, judgement, score_delta):
        mapping = {
            "PERFECT": 2.0, "GREAT": 1.0, "GOOD": 0.4,
            "ALMOST": 0.1, "BOO": -0.2, "MISS": -1.0,
        }
        return mapping.get(judgement, 0.0)

    def _model_path(self, song_name):
        os.makedirs("models", exist_ok=True)
        safe = song_name.replace(" ", "_")
        return os.path.join("models", f"ddr_{safe}.pkl")

    def save_model(self, song_name):
        path = self._model_path(song_name)
        with open(path, "wb") as f:
            pickle.dump({
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "Q": dict(self.Q),
                "total_episodes": self.total_episodes,
                "training_scores": self.training_scores,
            }, f)
        print(f"💾 Modelo guardado en {path}")

    def load_model(self, song_name):
        path = self._model_path(song_name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)
            self.epsilon = data.get("epsilon", self.epsilon)
            self.Q = defaultdict(float, data.get("Q", {}))
            self.total_episodes = data.get("total_episodes", 0)
            self.training_scores = data.get("training_scores", [])
            print(f"📂 Modelo cargado desde {path}")
        else:
            print("ℹ️ No hay modelo previo, comenzando desde cero.")


# === SETTINGS ================================================================
SCREEN_WIDTH, SCREEN_HEIGHT, FPS = 1280, 720, 60
WHITE, PINK, PURPLE, BLACK = (255, 255, 255), (255, 105, 180), (170, 0, 255), (0, 0, 0)
BASE_ARROW_SPEED = 5
COLUMN_X = {"left": 200, "down": 300, "up": 400, "right": 500}
HIT_ZONE_Y = 100


# === CLASES ==================================================================
class Arrow:
    def __init__(self, direction, image):
        self.direction = direction
        self.x = COLUMN_X[direction]
        self.y = SCREEN_HEIGHT + 50
        self.hit = False
        self.base_img = image
        self.image = self.rotated_img()

    def rotated_img(self):
        if self.direction == "left":
            return pygame.transform.rotate(self.base_img, 90)
        elif self.direction == "right":
            return pygame.transform.rotate(self.base_img, -90)
        elif self.direction == "down":
            return pygame.transform.rotate(self.base_img, 180)
        return self.base_img

    def update(self):
        self.y -= BASE_ARROW_SPEED

    def draw(self, screen):
        rect = self.image.get_rect(center=(self.x, self.y))
        screen.blit(self.image, rect)


# === MAIN GAME ===============================================================
class RhythmGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.joystick.init()
        self.agent_mode = True
        self.agent = DDRAgent(alpha=0.15, gamma=0.9, epsilon=0.05)  # modo inferencia

        # Joysticks
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        for joy in self.joysticks:
            joy.init()
            print(f"🎮 Detected joystick: {joy.get_name()}")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SCALED)
        pygame.display.set_caption("Dance Dance ReMixed")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Comic Sans MS", 36, bold=True)

        self.bg_start = pygame.transform.smoothscale(
            pygame.image.load("STARTBACKGROUND.png").convert(), (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        self.bg_select = pygame.transform.smoothscale(
            pygame.image.load("MUSICSELECT.png").convert(), (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.arrow_img = pygame.transform.scale(
            pygame.image.load("arrow.png").convert_alpha(), (90, 90)
        )
        self.arrowfill_img = pygame.transform.scale(
            pygame.image.load("arrowfill.png").convert_alpha(), (90, 90)
        )

        self.video = cv2.VideoCapture("butterfly_video.mp4")
        self.arrows, self.score, self.judgement = [], 0, ""
        self.current_scene, self.selected_song = "start", None
        self.chart_data, self.chart_index, self.song_start_time = [], 0, 0

    # --- Método para el agente ---
    def get_state_for_agent(self):
        dmin = {d: float("inf") for d in ["left", "down", "up", "right"]}
        for a in self.arrows:
            if not a.hit:
                dist = abs(a.y - HIT_ZONE_Y)
                if dist < dmin[a.direction]:
                    dmin[a.direction] = dist

        def disc(x):
            if x == float("inf"): return "none"
            if x < 10: return "perfect"
            if x < 25: return "great"
            if x < 50: return "good"
            if x < 70: return "almost"
            return "far"

        return tuple(disc(dmin[d]) for d in ["left", "down", "up", "right"])

    # --- (resto de tus métodos start_screen, song_select_screen, etc) ---
    # (idénticos a los que ya tienes, sin cambios excepto lo siguiente)
    # --- START SCREEN ---
    def start_screen(self):
        blink_interval = 0.6
        show_text = (time.time() % (blink_interval * 2)) < blink_interval
        self.screen.blit(self.bg_start, (0, 0))
        if show_text:
            prompt = self.font.render("Press ENTER to Start", True, WHITE)
            self.screen.blit(prompt, (SCREEN_WIDTH // 2 - 180, SCREEN_HEIGHT // 2 + 225))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.current_scene = "select"
        return True

    # --- SONG SELECT ---
    def song_select_screen(self):
        self.screen.blit(self.bg_select, (0, 0))
        songs = [
            "SMILE.dk - Butterfly",
            "DJ Simon - 321 STARS",
            "dj TAKA feat. NORIA - Love Love Sugar",
            "Celeste - Mirror Dance"
        ]
        if not hasattr(self, "song_index"):
            self.song_index = 0
        start_y = 130
        spacing = 130
        x = SCREEN_WIDTH // 2 + 20
        for i, song in enumerate(songs):
            if i == self.song_index:
                rect_color = (255, 182, 193)
                text_color = WHITE
            else:
                rect_color = (230, 230, 230)
                text_color = (120, 120, 120)
            rect = pygame.Rect(x - 20, start_y + i * spacing - 10, 500, 60)
            pygame.draw.rect(self.screen, rect_color, rect, border_radius=15)
            text = self.font.render(song, True, text_color)
            text_rect = text.get_rect(center=(890, start_y + i * spacing + 20))
            self.screen.blit(text, text_rect)
        prompt = self.font.render("Use ↑ ↓ to select, Enter to confirm", True, (180, 120, 255))
        self.screen.blit(prompt, (SCREEN_WIDTH // 2 - 230, SCREEN_HEIGHT - 100))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.song_index = (self.song_index - 1) % len(songs)
                elif event.key == pygame.K_DOWN:
                    self.song_index = (self.song_index + 1) % len(songs)
                elif event.key == pygame.K_RETURN:
                    selected = self.song_index
                    if selected == 0:
                        self.selected_song = "feel_the_power"
                    elif selected == 1:
                        self.selected_song = "321stars"
                    elif selected == 2:
                        self.selected_song = "lovelovesugar"
                    elif selected == 3:
                        self.selected_song = "mirror_dance"
                    self.current_scene = "game"
        return True

    # --- VIDEO ---
    def draw_video_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video.read()
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(frame_surface, (0, 0))

    # --- CHART LOADING ---
    def load_chart(self, filename="butterfly.chart"):
        self.chart_data = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    time_str, direction = line.split(",")
                    t = float(time_str.strip())
                    direction = direction.strip().lower()
                    self.chart_data.append((t, direction))
                except ValueError:
                    pass
        self.chart_index = 0
        print(f"Loaded {len(self.chart_data)} notes from chart.")

    # --- INPUT HANDLING ---
    def handle_input(self, direction):
        for arrow in self.arrows:
            if arrow.direction == direction and not arrow.hit:
                diff = abs(arrow.y - HIT_ZONE_Y)
                if diff < 10:
                    self.judgement = "PERFECT"
                    self.score += 100
                elif diff < 25:
                    self.judgement = "GREAT"
                    self.score += 50
                elif diff < 50:
                    self.judgement = "GOOD"
                    self.score += 20
                elif diff < 70:
                    self.judgement = "ALMOST"
                    self.score += 10
                else:
                    self.judgement = "BOO"
                arrow.hit = True
                return

    # --- JOYSTICK TO ARROW MAPPING ---
    def check_joystick(self):
        for joy in self.joysticks:
            x = joy.get_axis(0)
            y = joy.get_axis(1)
            hat_x, hat_y = 0, 0
            if joy.get_numhats() > 0:
                hat_x, hat_y = joy.get_hat(0)
            # Prioritize D-pad over analog
            dx = hat_x if hat_x != 0 else x
            dy = hat_y if hat_y != 0 else y
            if dx < -0.5:
                self.handle_input("left")
            elif dx > 0.5:
                self.handle_input("right")
            elif dy > 0.5:
                self.handle_input("down")
            elif dy < -0.5:
                self.handle_input("up")

    def render_hit_zone(self):
        for direction in ["left", "down", "up", "right"]:
            img = Arrow(direction, self.arrowfill_img).image
            rect = img.get_rect(center=(COLUMN_X[direction], HIT_ZONE_Y))
            self.screen.blit(img, rect)

    # --- GAME LOOP ---
    def game_loop(self):
        # (config de canción igual que tu versión)
        # ... [mismo código de carga de chart/video/música] ...
        self.load_chart("butterfly.chart")
        pygame.mixer.music.load("butterfly_recording.mp3")
        pygame.mixer.music.play()
        self.video = cv2.VideoCapture("butterfly_video.mp4")
        self.song_start_time = time.time()

        running = True
        while running:
            self.clock.tick(FPS)
            now = pygame.mixer.music.get_pos() / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT: self.handle_input("left")
                    elif event.key == pygame.K_RIGHT: self.handle_input("right")
                    elif event.key == pygame.K_UP: self.handle_input("up")
                    elif event.key == pygame.K_DOWN: self.handle_input("down")

            self.check_joystick()

            # === Acción automática del agente ===
            if self.agent_mode:
                state = self.get_state_for_agent()
                action = self.agent.choose_action(state)
                if action != "none":
                    self.handle_input(action)

            while self.chart_index < len(self.chart_data) and now >= self.chart_data[self.chart_index][0]:
                _, direction = self.chart_data[self.chart_index]
                self.arrows.append(Arrow(direction, self.arrow_img))
                self.chart_index += 1

            for arrow in self.arrows:
                arrow.update()
            self.arrows = [a for a in self.arrows if a.y > -50 and not a.hit]

            self.draw_video_frame()
            self.render_hit_zone()
            for arrow in self.arrows:
                arrow.draw(self.screen)

            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_text, (20, 20))
            if self.judgement:
                j_text = self.font.render(self.judgement, True, PINK)
                self.screen.blit(j_text, (SCREEN_WIDTH // 2 - 80, HIT_ZONE_Y + 40))
            pygame.display.flip()

            if not pygame.mixer.music.get_busy():
                running = False

            # ...

    # (resto igual a tu código original)
    # --- RUN ---
    def run(self):
        running = True
        while running:
            if self.current_scene == "start":
                running = self.start_screen()
            elif self.current_scene == "select":
                running = self.song_select_screen()
            elif self.current_scene == "game":
                self.game_loop()
                running = False
        pygame.quit()


# --- RUN ---
if __name__ == "__main__":
    RhythmGame().run()
