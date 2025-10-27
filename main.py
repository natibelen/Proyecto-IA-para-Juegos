import numpy as np
import pygame
import cv2
import mss
import mss.tools
import threading
import time
from PIL import Image
from agent import get_region, check_pixel

# --- SETTINGS ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60


latest_frame = None
stop_capture = False

# Colors
WHITE = (255, 255, 255)
PINK = (255, 105, 180)
PURPLE = (170, 0, 255)
BLACK = (0, 0, 0)

# Arrow settings
BASE_ARROW_SPEED = 5  # default speed
COLUMN_X = {
    "left": 200,
    "down": 300,
    "up": 400,
    "right": 500,
}
HIT_ZONE_Y = 100

# --- CLASSES ---
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
        else:  # up
            return self.base_img

    def update(self):
        self.y -= BASE_ARROW_SPEED

    def draw(self, screen):
        rect = self.image.get_rect(center=(self.x, self.y))
        screen.blit(self.image, rect)


# --- MAIN GAME ---
class RhythmGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.joystick.init()

        # Joystick setup
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        for joy in self.joysticks:
            joy.init()
            print(f"🎮 Detected joystick: {joy.get_name()}")

        # Fullscreen mode
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SCALED)
        pygame.display.set_caption("Dance Dance ReMixed")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Comic Sans MS", 36, bold=True)

        # Backgrounds
        self.bg_start = pygame.image.load("STARTBACKGROUND.png").convert()
        self.bg_start = pygame.transform.smoothscale(self.bg_start, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.bg_select = pygame.image.load("MUSICSELECT.png").convert()
        self.bg_select = pygame.transform.smoothscale(self.bg_select, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Arrows
        self.arrow_img = pygame.image.load("arrow.png").convert_alpha()
        self.arrow_img = pygame.transform.scale(self.arrow_img, (90, 90))
        self.arrowfill_img = pygame.image.load("arrowfill.png").convert_alpha()
        self.arrowfill_img = pygame.transform.scale(self.arrowfill_img, (90, 90))

        # Video placeholder
        self.video = cv2.VideoCapture("butterfly_video.mp4")

        # Game vars
        self.arrows = []
        self.score = 0
        self.judgement = ""
        self.current_scene = "start"
        self.selected_song = None
        self.chart_data = []
        self.chart_index = 0
        self.song_start_time = 0

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
        if self.selected_song == "feel_the_power":
            chart_file = "butterfly.chart"
            music_file = "butterfly_recording.mp3"
            video_file = "love love sugar.mp4"
        elif self.selected_song == "321stars":
            chart_file = "321stars.chart"
            music_file = "DJ SIMON - 321STARS (HQ) [K2l7HXC0p1c].mp3"
            video_file = "love love sugar.mp4"
        elif self.selected_song == "lovelovesugar":
            chart_file = "lovelovesugar.chart"
            music_file = "dj TAKA feat. NORIA - LOVE LOVE SUGAR (HQ).mp3"
            video_file = "love love sugar.mp4"
        elif self.selected_song == "mirror_dance":
            chart_file = "mirror_dance.chart"
            music_file = "mirror_dance.mp3"
            video_file = "love love sugar.mp4"
        else:
            chart_file = "butterfly.chart"
            music_file = "butterfly_recording.mp3"
            video_file = "love love sugar.mp4"

        if self.selected_song == "321stars":
            arrow_speed = 15  # change this to test speed
        else:
            arrow_speed = BASE_ARROW_SPEED

        # Load assets
        self.load_chart(chart_file)
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
        self.video = cv2.VideoCapture(video_file)
        self.song_start_time = time.time()

        running = True
        while running:
            self.clock.tick(FPS)
            now = (pygame.mixer.music.get_pos() / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_LEFT:
                        self.handle_input("left")
                    elif event.key == pygame.K_RIGHT:
                        self.handle_input("right")
                    elif event.key == pygame.K_UP:
                        self.handle_input("up")
                    elif event.key == pygame.K_DOWN:
                        self.handle_input("down")

            self.check_joystick()

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


    def game_loop_agent(self):
        if self.selected_song == "feel_the_power":
            chart_file = "butterfly.chart"
            music_file = "butterfly_recording.mp3"
            video_file = "love love sugar.mp4"
        elif self.selected_song == "321stars":
            chart_file = "321stars.chart"
            music_file = "DJ SIMON - 321STARS (HQ) [K2l7HXC0p1c].mp3"
            video_file = "love love sugar.mp4"
        elif self.selected_song == "lovelovesugar":
            chart_file = "lovelovesugar.chart"
            music_file = "dj TAKA feat. NORIA - LOVE LOVE SUGAR (HQ).mp3"
            video_file = "love love sugar.mp4"
        elif self.selected_song == "mirror_dance":
            chart_file = "mirror_dance.chart"
            music_file = "mirror_dance.mp3"
            video_file = "love love sugar.mp4"
        else:
            chart_file = "butterfly.chart"
            music_file = "butterfly_recording.mp3"
            video_file = "love love sugar.mp4"

        if self.selected_song == "321stars":
            arrow_speed = 15  # change this to test speed
        else:
            arrow_speed = BASE_ARROW_SPEED

        # Load assets
        self.load_chart(chart_file)
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
        self.video = cv2.VideoCapture(video_file)
        self.song_start_time = time.time()

        running = True

        capture_thread = threading.Thread(target=take_screenshot, daemon=True)
        capture_thread.start()

        while running:

            if latest_frame is not None:
                if check_pixel(latest_frame, 62, 40):
                    self.handle_input("left")
                elif check_pixel(latest_frame, 128, 4):
                    self.handle_input("down")
                elif check_pixel(latest_frame, 228, 73):
                    self.handle_input("up")
                elif check_pixel(latest_frame, 292, 40):
                    self.handle_input("right")


            self.clock.tick(FPS)
            now = (pygame.mixer.music.get_pos() / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False


            self.check_joystick()

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

    # --- MAIN LOOP ---
    def run(self):
        running = True
        while running:
            if self.current_scene == "start":
                running = self.start_screen()
            elif self.current_scene == "select":
                running = self.song_select_screen()
            elif self.current_scene == "game":
                self.game_loop_agent()
                running = False
        pygame.quit()


def take_screenshot():
    
    global latest_frame
    sct = mss.mss()
    while not stop_capture:
        region = get_region("Dance Dance ReMixed")
        img = np.array(sct.grab(region))
        latest_frame = img


# --- RUN ---
if __name__ == "__main__":
    RhythmGame().run()
