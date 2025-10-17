import pygame
import random
import time
import cv2

# --- SETTINGS ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Colors
WHITE = (255, 255, 255)
PINK = (255, 105, 180)
PURPLE = (170, 0, 255)
BLACK = (0, 0, 0)

# Arrow settings
ARROW_SPEED = 5
COLUMN_X = {
    "left": 200,
    "down": 300,
    "up": 400,
    "right": 500,
}
HIT_ZONE_Y = 100
BPM = 138
SPAWN_INTERVAL = 60.0 / BPM


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
        self.y -= ARROW_SPEED

    def draw(self, screen):
        rect = self.image.get_rect(center=(self.x, self.y))
        screen.blit(self.image, rect)


# --- MAIN GAME ---
class RhythmGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

        # Fullscreen mode (keeps aspect ratio nicely on any screen)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SCALED)
        pygame.display.set_caption("Dance Dance ReMixed")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Comic Sans MS", 36, bold=True)

        # === Load and scale backgrounds ===
        # Start screen background
        self.bg_start = pygame.image.load("STARTBACKGROUND.png").convert()
        self.bg_start = pygame.transform.smoothscale(self.bg_start, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Song select background (can be replaced later)
        self.bg_select = pygame.image.load("STARTBACKGROUND.png").convert()
        self.bg_select = pygame.transform.smoothscale(self.bg_select, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.arrow_img = pygame.image.load("arrow.png").convert_alpha()
        self.arrow_img = pygame.transform.scale(self.arrow_img, (70, 70))

        # Video setup
        self.video = cv2.VideoCapture("feel_the_power_video.mp4")

        # Game vars
        self.arrows = []
        self.last_spawn = time.time()
        self.score = 0
        self.judgement = ""
        self.current_scene = "start"  # start -> select -> game
        self.selected_song = None

    # --- 🌸 START SCREEN ---
    def start_screen(self):
        self.screen.blit(self.bg_start, (0, 0))
        title = self.font.render("Dance Dance Remix", True, WHITE)
        prompt = self.font.render("Press ENTER to Start", True, PINK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 100))
        self.screen.blit(prompt, (SCREEN_WIDTH // 2 - 180, SCREEN_HEIGHT // 2 + 50))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.current_scene = "select"
        return True

    # --- 💿 SONG SELECT SCREEN ---
    def song_select_screen(self):
        self.screen.blit(self.bg_select, (0, 0))
        title = self.font.render("Select a Song 🎶", True, WHITE)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - 150, 100))

        # Song options (you can add art/buttons later!)
        songs = ["Bratz - Feel The Power", "DJ Simon - 321 STARS"]
        for i, song in enumerate(songs):
            color = PINK if i == 0 else WHITE
            text = self.font.render(song, True, color)
            self.screen.blit(text, (SCREEN_WIDTH // 2 - 250, 250 + i * 80))

        prompt = self.font.render("Press 1 or 2 to Select", True, PURPLE)
        self.screen.blit(prompt, (SCREEN_WIDTH // 2 - 200, 500))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.selected_song = "feel_the_power"
                    self.current_scene = "game"
                elif event.key == pygame.K_2:
                    self.selected_song = "321stars"
                    self.current_scene = "game"
        return True

    # --- 🎥 GAMEPLAY SCREEN ---
    def draw_video_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video.read()
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(frame_surface, (0, 0))

    def spawn_arrow(self):
        direction = random.choice(["left", "down", "up", "right"])
        self.arrows.append(Arrow(direction, self.arrow_img))

    def handle_input(self, key):
        key_map = {
            pygame.K_LEFT: "left",
            pygame.K_DOWN: "down",
            pygame.K_UP: "up",
            pygame.K_RIGHT: "right",
        }

        if key not in key_map:
            return
        direction = key_map[key]
        for arrow in self.arrows:
            if arrow.direction == direction and not arrow.hit:
                diff = abs(arrow.y - HIT_ZONE_Y)
                if diff < 10:
                    self.judgement = "Perfect!"
                    self.score += 100
                elif diff < 25:
                    self.judgement = "Good!"
                    self.score += 50
                elif diff < 40:
                    self.judgement = "Almost!"
                    self.score += 20
                else:
                    self.judgement = "Miss"
                arrow.hit = True
                return

    def render_hit_zone(self):
        pygame.draw.rect(self.screen, PURPLE, (100, HIT_ZONE_Y - 10, 450, 20), 3)
        for direction in ["left", "down", "up", "right"]:
            img = Arrow(direction, self.arrow_img).image
            rect = img.get_rect(center=(COLUMN_X[direction], HIT_ZONE_Y))
            self.screen.blit(img, rect)

    def game_loop(self):
        pygame.mixer.music.load("Bratz - Feel The Power.mp3")
        pygame.mixer.music.play()
        running = True

        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        self.handle_input(event.key)

            if time.time() - self.last_spawn >= SPAWN_INTERVAL:
                self.spawn_arrow()
                self.last_spawn = time.time()

            for arrow in self.arrows:
                arrow.update()

            self.arrows = [a for a in self.arrows if a.y > -50]

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
                self.game_loop()
                running = False
        pygame.quit()


# --- RUN ---
if __name__ == "__main__":
    RhythmGame().run()
