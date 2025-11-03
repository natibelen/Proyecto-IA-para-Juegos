import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# ============================================
# CONFIGURACIÓN MEJORADA
# ============================================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60  # ¡CAMBIO 1: FPS más bajo = más fácil de aprender!

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (100, 100, 255)

BASE_ARROW_SPEED = 5
COLUMN_X = {"left": 200, "down": 300, "up": 400, "right": 500}
HIT_ZONE_Y = 100

# ============================================
# CLASE ARROW
# ============================================
class Arrow:
    def __init__(self, direction, y_position=SCREEN_HEIGHT + 50):
        self.direction = direction
        self.x = COLUMN_X[direction]
        self.y = y_position
        self.hit = False
        self.spawn_time = 0  # Para calcular cuánto tiempo lleva en pantalla

    def update(self):
        self.y -= BASE_ARROW_SPEED

    def draw(self, screen, font):
        color = GREEN if self.hit else WHITE
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 30)
        text = font.render(self.direction[0].upper(), True, BLACK)
        text_rect = text.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(text, text_rect)


# ============================================
# RED NEURONAL DQN (MÁS GRANDE)
# ============================================
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        # CAMBIO 2: Red más grande para capturar patrones complejos
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, action_size)
        self.dropout = nn.Dropout(0.2)  # Prevenir overfitting
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


# ============================================
# AGENTE DQN MEJORADO
# ============================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # CAMBIO 3: Más memoria
        self.gamma = 0.95  # CAMBIO 4: Gamma más bajo = valora más el presente
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # CAMBIO 5: Más exploración mínima
        self.epsilon_decay = 0.9985  # CAMBIO 6: Decay más lento
        self.learning_rate = 0.0001  # CAMBIO 7: Learning rate más bajo
        
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # CAMBIO 8: Huber loss (más estable)
        
        # Para priorizar experiencias importantes
        self.priorities = deque(maxlen=50000)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # CAMBIO 9: Priorizar experiencias con recompensa alta
        priority = abs(reward) + 1
        self.priorities.append(priority)
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def replay(self, batch_size=128):  # CAMBIO 10: Batch más grande
        if len(self.memory) < batch_size:
            return 0
        
        # Muestreo con prioridad
        priorities = np.array(list(self.priorities))
        priorities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=priorities)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # CAMBIO 11: Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================
# ENTORNO MEJORADO
# ============================================
class RhythmGameRL:
    def __init__(self, chart_file="lovelovesugar.chart", visualize=True):
        pygame.init()
        self.visualize = visualize
        
        if visualize:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("DQN Training MEJORADO - Love Love Sugar")
            self.font = pygame.font.SysFont("Arial", 20)
            self.font_big = pygame.font.SysFont("Arial", 32, bold=True)
            self.clock = pygame.time.Clock()
        
        self.chart_data = []
        self.load_chart(chart_file)
        self.reset()
        
    def load_chart(self, filename):
        try:
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
            print(f"\n{'='*70}")
            print(f"CHART CARGADO: {filename}")
            print(f"Total de notas: {len(self.chart_data)}")
            print(f"Duracion: ~{self.chart_data[-1][0]:.1f} segundos")
            print(f"{'='*70}\n")
        except FileNotFoundError:
            print(f"[ERROR] No se encontro {filename}")
            exit(1)
    
    def reset(self):
        self.arrows = []
        self.score = 0
        self.chart_index = 0
        self.time = 0
        self.done = False
        self.judgement = ""
        self.judgement_timer = 0
        self.perfect_hits = 0
        self.great_hits = 0
        self.good_hits = 0
        self.missed_arrows = 0
        self.steps = 0
        self.total_reward = 0  # Para debug
        return self.get_state()
    
    def get_state(self):
        """
        CAMBIO 12: Estado mejorado con MÁS información
        
        Antes: Solo posición de 8 flechas
        Ahora: 
          - 10 flechas más cercanas (más contexto)
          - Distancia Y velocidad de cada flecha
          - Tiempo desde última nota golpeada
          - Progreso en la canción
        """
        state = []
        
        # Obtener las 10 flechas más cercanas
        active_arrows = sorted(
            [a for a in self.arrows if not a.hit and a.y > -50],
            key=lambda a: abs(a.y - HIT_ZONE_Y)
        )[:10]  # Aumentado de 8 a 10
        
        for arrow in active_arrows:
            # One-hot encoding de dirección (4 valores)
            direction_encoding = [0, 0, 0, 0]
            dir_map = {"left": 0, "down": 1, "up": 2, "right": 3}
            direction_encoding[dir_map[arrow.direction]] = 1
            
            # Distancia normalizada
            distance = (arrow.y - HIT_ZONE_Y) / SCREEN_HEIGHT
            
            # NUEVO: Velocidad relativa (qué tan rápido se acerca)
            velocity = BASE_ARROW_SPEED / SCREEN_HEIGHT
            
            # 4 (dirección) + 1 (distancia) + 1 (velocidad) = 6 features por flecha
            state.extend(direction_encoding + [distance, velocity])
        
        # Rellenar con ceros
        while len(state) < 60:  # 10 flechas * 6 features
            state.append(0)
        
        # NUEVO: Contexto adicional
        # Progreso en la canción (0-1)
        progress = self.chart_index / len(self.chart_data) if self.chart_data else 0
        state.append(progress)
        
        # Tiempo normalizado
        max_time = self.chart_data[-1][0] if self.chart_data else 100
        normalized_time = min(1.0, self.time / max_time)
        state.append(normalized_time)
        
        # Total: 60 + 2 = 62 features
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        CAMBIO 13: Sistema de recompensas mejorado (Reward Shaping)
        """
        reward = 0
        self.judgement = ""
        self.judgement_timer = max(0, self.judgement_timer - 1)
        
        if action < 4:
            direction = ["left", "down", "up", "right"][action]
            hit_reward, hit_type = self.handle_input(direction)
            reward += hit_reward
            
            # NUEVO: Penalización menor por presionar cuando no hay flecha
            # (antes era -5, ahora -2)
            if hit_type == "empty":
                reward = -2
        
        # NUEVO: Pequeña recompensa por sobrevivir (encourage exploration)
        reward += 0.1
        
        self.time += 1.0 / FPS
        self.steps += 1
        
        self.spawn_arrows()
        
        for arrow in self.arrows:
            arrow.update()
        
        # Detectar misses con RECOMPENSAS GRADUALES
        before_arrows = [a for a in self.arrows if not a.hit and a.y > HIT_ZONE_Y]
        self.arrows = [a for a in self.arrows if a.y > -100]
        after_arrows = [a for a in self.arrows if not a.hit and a.y > HIT_ZONE_Y]
        
        # Flechas que pasaron el hit zone
        for arrow in before_arrows:
            if arrow not in after_arrows and not arrow.hit:
                # CAMBIO 14: Penalización más suave por miss
                miss_distance = abs(arrow.y - HIT_ZONE_Y)
                if miss_distance < 50:
                    penalty = -15  # Estuvo cerca
                else:
                    penalty = -25  # Muy lejos
                reward += penalty
                self.missed_arrows += 1
        
        # Terminar cuando se acaban las notas
        if self.chart_index >= len(self.chart_data) and len(self.arrows) == 0:
            self.done = True
            # CAMBIO 15: Bonus por completar proporcional al score
            completion_bonus = 100 + (self.score / 10)
            reward += completion_bonus
        
        # Safety timeout
        if self.steps > 20000:
            self.done = True
            reward -= 50  # Penalización por timeout
        
        self.total_reward += reward
        next_state = self.get_state()
        return next_state, reward, self.done
    
    def spawn_arrows(self):
        while self.chart_index < len(self.chart_data):
            note_time, note_direction = self.chart_data[self.chart_index]
            travel_time = (SCREEN_HEIGHT + 50 - HIT_ZONE_Y) / BASE_ARROW_SPEED * (1.0 / FPS)
            spawn_time = note_time - travel_time
            
            if self.time >= spawn_time:
                arrow = Arrow(note_direction)
                arrow.spawn_time = self.time
                self.arrows.append(arrow)
                self.chart_index += 1
            else:
                break
    
    def handle_input(self, direction):
        """
        CAMBIO 16: Sistema de puntuación más generoso
        """
        candidates = [a for a in self.arrows 
                     if a.direction == direction and not a.hit]
        
        if not candidates:
            return -2, "empty"  # Penalización reducida
        
        closest = min(candidates, key=lambda a: abs(a.y - HIT_ZONE_Y))
        diff = abs(closest.y - HIT_ZONE_Y)
        
        # Umbrales más generosos
        if diff < 15:  # Era 10
            self.judgement = "PERFECT"
            reward = 100
            self.perfect_hits += 1
            self.score += 100
            hit_type = "perfect"
        elif diff < 35:  # Era 25
            self.judgement = "GREAT"
            reward = 60  # Era 50
            self.great_hits += 1
            self.score += 60
            hit_type = "great"
        elif diff < 60:  # Era 50
            self.judgement = "GOOD"
            reward = 30  # Era 20
            self.good_hits += 1
            self.score += 30
            hit_type = "good"
        elif diff < 100:  # Era 70
            self.judgement = "OK"
            reward = 10
            self.score += 10
            hit_type = "ok"
        else:
            self.judgement = "BOO"
            reward = -10  # Era -5
            hit_type = "boo"
        
        closest.hit = True
        self.judgement_timer = 15  # Mostrar por 15 frames
        return reward, hit_type
    
    def render(self, episode, epsilon, avg_score, best_score):
        if not self.visualize:
            return
        
        self.screen.fill(BLACK)
        
        # Hit zones con efecto
        for direction in ["left", "down", "up", "right"]:
            pygame.draw.circle(self.screen, GREEN, 
                             (COLUMN_X[direction], HIT_ZONE_Y), 45, 3)
            pygame.draw.circle(self.screen, (0, 100, 0), 
                             (COLUMN_X[direction], HIT_ZONE_Y), 35, 1)
        
        # Flechas
        for arrow in self.arrows:
            arrow.draw(self.screen, self.font)
        
        # Panel de información
        y = 15
        info = [
            ("ENTRENAMIENTO MEJORADO", WHITE, self.font_big),
            ("", WHITE, self.font),
            (f"Episodio: {episode}/500", YELLOW, self.font),
            (f"Exploracion: {epsilon:.1%}", BLUE, self.font),
            ("", WHITE, self.font),
            (f"Score: {self.score}", WHITE, self.font),
            (f"Promedio: {avg_score:.0f}", WHITE, self.font),
            (f"Mejor: {best_score}", GREEN, self.font),
            ("", WHITE, self.font),
            (f"Perfect: {self.perfect_hits}", GREEN, self.font),
            (f"Great: {self.great_hits}", YELLOW, self.font),
            (f"Good: {self.good_hits}", WHITE, self.font),
            (f"Missed: {self.missed_arrows}", RED, self.font),
            ("", WHITE, self.font),
            (f"Reward Total: {self.total_reward:.0f}", BLUE, self.font),
        ]
        
        for text_str, color, font in info:
            text = font.render(text_str, True, color)
            self.screen.blit(text, (15, y))
            y += font.get_height() + 3
        
        # Judgement
        if self.judgement_timer > 0:
            colors = {
                "PERFECT": GREEN,
                "GREAT": YELLOW,
                "GOOD": WHITE,
                "OK": (255, 165, 0),
                "BOO": RED
            }
            color = colors.get(self.judgement, WHITE)
            text = self.font_big.render(self.judgement, True, color)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, HIT_ZONE_Y + 70))
            
            # Efecto de sombra
            shadow = self.font_big.render(self.judgement, True, BLACK)
            shadow_rect = text_rect.copy()
            shadow_rect.x += 2
            shadow_rect.y += 2
            self.screen.blit(shadow, shadow_rect)
            self.screen.blit(text, text_rect)
        
        # Barra de progreso
        progress = self.chart_index / len(self.chart_data)
        bar_w, bar_h = 500, 25
        bar_x = SCREEN_WIDTH - bar_w - 20
        bar_y = SCREEN_HEIGHT - 50
        
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 2)
        pygame.draw.rect(self.screen, GREEN, (bar_x+2, bar_y+2, int((bar_w-4) * progress), bar_h-4))
        
        prog_text = self.font.render(f"Progreso: {progress:.0%} ({self.chart_index}/{len(self.chart_data)})", True, WHITE)
        self.screen.blit(prog_text, (bar_x, bar_y - 25))
        
        pygame.display.flip()
        self.clock.tick(FPS)


# ============================================
# ENTRENAMIENTO MEJORADO
# ============================================
def train_improved():
    print("\n" + "="*70)
    print("ENTRENAMIENTO DQN MEJORADO - LOVE LOVE SUGAR")
    print("="*70)
    print("\nMEJORAS IMPLEMENTADAS:")
    print("  [1] FPS reducido: 200 -> 60 (timing más fácil)")
    print("  [2] Red neuronal más grande: 256 -> 512 neuronas")
    print("  [3] Estado más rico: 40 -> 62 features")
    print("  [4] Recompensas graduales (reward shaping)")
    print("  [5] Epsilon decay más lento: mantiene exploración")
    print("  [6] Umbrales de hit más generosos")
    print("  [7] Muestreo prioritizado de experiencias")
    print("  [8] Gradient clipping + Huber loss")
    print("  [9] Batch size más grande: 64 -> 128")
    print("  [10] Más episodios: 300 -> 500")
    print("\nPresiona Ctrl+C para detener")
    print("="*70 + "\n")
    
    STATE_SIZE = 62  # Aumentado
    ACTION_SIZE = 5
    EPISODES = 500  # Más episodios
    
    env = RhythmGameRL(chart_file="lovelovesugar.chart", visualize=True)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    
    scores = []
    avg_scores = []
    losses = []
    best_score = 0
    
    start_time = time.time()
    
    try:
        for episode in range(EPISODES):
            episode_start = time.time()
            state = env.reset()
            episode_losses = []
            
            while not env.done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                action = agent.act(state, training=True)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                # Entrenar más frecuentemente
                if len(agent.memory) >= 128:
                    loss = agent.replay(batch_size=128)
                    if loss > 0:
                        episode_losses.append(loss)
                
                state = next_state
                
                # Visualizar cada 3 episodios
                if episode % 3 == 0:
                    avg_score = np.mean(scores[-100:]) if scores else 0
                    env.render(episode, agent.epsilon, avg_score, best_score)
            
            # Actualizar target cada 5 episodios (más frecuente)
            if episode % 5 == 0:
                agent.update_target_model()
            
            agent.decay_epsilon()
            
            scores.append(env.score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            losses.append(avg_loss)
            
            if env.score > best_score:
                best_score = env.score
                # Guardar mejor modelo
                torch.save(agent.model.state_dict(), "dqn_lovelovesugar_best.pth")
            
            episode_time = time.time() - episode_start
            elapsed = time.time() - start_time
            eta = (elapsed / (episode + 1)) * (EPISODES - episode - 1)
            
            total_notes = env.perfect_hits + env.great_hits + env.good_hits + env.missed_arrows
            accuracy = ((env.perfect_hits + env.great_hits + env.good_hits) / total_notes * 100) if total_notes > 0 else 0
            
            print(f"\n{'='*70}")
            print(f"EPISODIO {episode + 1}/{EPISODES}")
            print(f"{'='*70}")
            print(f"Score:        {env.score:5d}  (Avg: {avg_score:6.1f}, Best: {best_score})")
            print(f"Hits:         P:{env.perfect_hits:3d} G:{env.great_hits:3d} O:{env.good_hits:3d}")
            print(f"Missed:       {env.missed_arrows:3d} ({100-accuracy:.1f}%)")
            print(f"Accuracy:     {accuracy:.1f}%")
            print(f"Total Reward: {env.total_reward:.1f}")
            print(f"Epsilon:      {agent.epsilon:.3f} ({agent.epsilon*100:.1f}% random)")
            print(f"Loss:         {avg_loss:.4f}")
            print(f"Memoria:      {len(agent.memory)} experiencias")
            print(f"Tiempo:       {episode_time:.1f}s/ep  |  ETA: {eta/60:.1f} min")
            
            if episode > 0 and episode % 10 == 0:
                recent_avg = np.mean(scores[max(0, episode-10):episode])
                old_avg = np.mean(scores[:10]) if episode >= 10 else scores[0]
                improvement = recent_avg - old_avg
                print(f"Mejora:       {improvement:+.1f} puntos (últimos 10 vs iniciales)")
            
            if episode % 100 == 0 and episode > 0:
                torch.save(agent.model.state_dict(), f"dqn_lovelovesugar_ep{episode}.pth")
                print(f"\n[GUARDADO] Checkpoint: dqn_lovelovesugar_ep{episode}.pth")
        
        print("\n" + "="*70)
        print("¡ENTRENAMIENTO COMPLETADO!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("ENTRENAMIENTO INTERRUMPIDO")
        print("="*70)
    
    finally:
        torch.save(agent.model.state_dict(), "dqn_lovelovesugar_final.pth")
        print(f"\n[GUARDADO] Modelo final guardado")
        
        print("\n" + "="*70)
        print("RESUMEN FINAL")
        print("="*70)
        print(f"Episodios:     {len(scores)}")
        print(f"Score Avg:     {np.mean(scores):.1f}")
        print(f"Score Mejor:   {best_score}")
        print(f"Score Final:   {scores[-1]}")
        print(f"Mejora:        {scores[-1] - scores[0]:+d}")
        
        if len(scores) >= 20:
            first_20 = np.mean(scores[:20])
            last_20 = np.mean(scores[-20:])
            print(f"Primeros 20:   {first_20:.1f}")
            print(f"Ultimos 20:    {last_20:.1f}")
            print(f"Diferencia:    {last_20 - first_20:+.1f}")
        
        plot_results(scores, avg_scores, losses)
        pygame.quit()


def plot_results(scores, avg_scores, losses):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scores
    axes[0, 0].plot(scores, alpha=0.5, label='Score', color='blue')
    axes[0, 0].plot(avg_scores, linewidth=2, label='Avg (100 ep)', color='red')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Score Evolution - IMPROVED')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(losses, color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Score distribution
    axes[1, 0].hist(scores, bins=40, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning curve
    if len(scores) >= 10:
        window = 10
        rolling_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
        axes[1, 1].plot(rolling_avg, color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rolling Average (10 ep)')
        axes[1, 1].set_title('Learning Progress')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_lovelovesugar_improved.png', dpi=150)
    print(f"\n[GRAFICAS] Guardadas en: training_lovelovesugar_improved.png")
    plt.show()


# ============================================
# EJECUCIÓN
# ============================================
if __name__ == "__main__":
    train_improved()