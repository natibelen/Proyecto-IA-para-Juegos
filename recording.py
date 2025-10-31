import pygame
import cv2
import time
import sys

# --- CONFIG ---
VIDEO_FILE = "321stars.mp4"
SONG_FILE = "music_files/DJ SIMON - 321STARS (HQ) [K2l7HXC0p1c].mp3"
CHART_FILE = "moveyourfeet.chart"
SCREEN_SIZE = (600, 700)
FPS = 30

# --- Initialize ---
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("🎵 Chart Recorder - 321STARS")

font = pygame.font.Font(None, 36)
WHITE = (255, 255, 255)

# --- Load arrows ---
base_arrow = pygame.image.load("resources/arrowfill.png").convert_alpha()
base_arrow = pygame.transform.scale(base_arrow, (80, 80))

arrow_images = {
    "up": base_arrow,
    "right": pygame.transform.rotate(base_arrow, -90),
    "down": pygame.transform.rotate(base_arrow, 180),
    "left": pygame.transform.rotate(base_arrow, 90),
}

arrow_positions = {
    "left": (SCREEN_SIZE[0] // 2 - 160, SCREEN_SIZE[1] - 150),
    "down": (SCREEN_SIZE[0] // 2 - 60, SCREEN_SIZE[1] - 150),
    "up": (SCREEN_SIZE[0] // 2 + 40, SCREEN_SIZE[1] - 150),
    "right": (SCREEN_SIZE[0] // 2 + 140, SCREEN_SIZE[1] - 150),
}

# --- Video setup ---
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    sys.exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / (video_fps if video_fps > 0 else FPS)

# --- Audio setup ---
pygame.mixer.music.load(SONG_FILE)
pygame.mixer.music.play()

# --- Joystick setup ---
pygame.joystick.init()
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"🎮 Joystick detected: {joystick.get_name()}")
    print(f"  → Axes: {joystick.get_numaxes()}")
    print(f"  → Buttons: {joystick.get_numbuttons()}")
    print(f"  → Hats: {joystick.get_numhats()}")
else:
    print("⚠️ No joystick found. Only keyboard input will work.")

chart_data = []
pressed = {}
start_time = time.time()
running = True

# --- Main loop ---
while running:
    current_time = time.time() - start_time
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, SCREEN_SIZE)
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(frame_surface, (0, 0))

    # --- Handle input ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Keyboard arrows
        elif event.type == pygame.KEYDOWN:
            now = time.time() - start_time
            if event.key == pygame.K_LEFT:
                chart_data.append((now, "left"))
                pressed["left"] = 0.3
            elif event.key == pygame.K_DOWN:
                chart_data.append((now, "down"))
                pressed["down"] = 0.3
            elif event.key == pygame.K_UP:
                chart_data.append((now, "up"))
                pressed["up"] = 0.3
            elif event.key == pygame.K_RIGHT:
                chart_data.append((now, "right"))
                pressed["right"] = 0.3

        # Joystick buttons (face buttons + PS4 D-pad buttons)
        elif event.type == pygame.JOYBUTTONDOWN:
            now = time.time() - start_time

            # Face buttons
            if event.button == 0:  # X
                chart_data.append((now, "down"))
                pressed["down"] = 0.3
            elif event.button == 1:  # O
                chart_data.append((now, "right"))
                pressed["right"] = 0.3
            elif event.button == 2:  # Square
                chart_data.append((now, "left"))
                pressed["left"] = 0.3
            elif event.button == 3:  # Triangle
                chart_data.append((now, "up"))
                pressed["up"] = 0.3

            # PS4 D-pad buttons (common mapping; check with test if different)
            elif event.button == 11:  # D-pad Up
                chart_data.append((now, "up"))
                pressed["up"] = 0.3
            elif event.button == 14:  # D-pad Right
                chart_data.append((now, "right"))
                pressed["right"] = 0.3
            elif event.button == 12:  # D-pad Down
                chart_data.append((now, "down"))
                pressed["down"] = 0.3
            elif event.button == 13:  # D-pad Left
                chart_data.append((now, "left"))
                pressed["left"] = 0.3

    # Draw pressed arrows
    for direction, t in list(pressed.items()):
        if t > 0:
            screen.blit(arrow_images[direction], arrow_positions[direction])
            pressed[direction] -= frame_time
        else:
            del pressed[direction]

    # Timer display
    time_text = font.render(f"Recording... {current_time:.1f}s", True, WHITE)
    screen.blit(time_text, (20, 20))

    pygame.display.flip()

    # Sync video playback to real time
    expected_time = start_time + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    sleep_time = expected_time - time.time()
    if sleep_time > 0:
        time.sleep(sleep_time)

# --- Save chart ---
pygame.mixer.music.stop()
cap.release()
pygame.quit()

with open(CHART_FILE, "w") as f:
    for t, direction in chart_data:
        f.write(f"{t:.2f}, {direction}\n")

print(f"✅ Chart saved to {CHART_FILE} with {len(chart_data)} notes.")
