
import pygame
import cv2
import time
import argparse
import numpy as np
from collections import defaultdict
import os
import pickle

# Importar del juego
from main import DDRAgent, Arrow, COLUMN_X, HIT_ZONE_Y, BASE_ARROW_SPEED

class AutoTrainer:
    """Entrenador automático para el agente DDR (simulación sin pygame por defecto)."""
    def __init__(self, song_name, visualize=False, alpha=0.15, gamma=0.9, epsilon_start=0.5):
        self.song_name = song_name
        self.visualize = visualize
        self.agent = DDRAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start)

        # Config de canciones (ajusta rutas si cambian los nombres de tus archivos)
        self.song_configs = {
            "321stars": {
                "chart": "321stars.chart",
                "music": "DJ SIMON - 321STARS (HQ) [K2l7HXC0p1c].mp3",
                "video": "321stars.mp4"
            },
            "feel_the_power": {
                "chart": "butterfly.chart",
                "music": "butterfly_recording.mp3",
                "video": "butterfly_video.mp4"
            },
            "lovelovesugar": {
                "chart": "lovelovesugar.chart",
                "music": "dj TAKA feat. NORIA - LOVE LOVE SUGAR (HQ).mp3",
                "video": "love love sugar.mp4"
            },
            "mirror_dance": {
                "chart": "mirror_dance.chart",
                "music": "mirror_dance.mp3",
                "video": "mirror_dance.mp4"
            }
        }
        if song_name not in self.song_configs:
            raise ValueError(f"Canción '{song_name}' no reconocida. Usa: {list(self.song_configs.keys())}")
        self.chart_data = self.load_chart(self.song_configs[song_name]["chart"])

        # Visualización opcional
        if self.visualize:
            pygame.init()
            pygame.mixer.init()
            self.screen = pygame.display.set_mode((1280, 720))
            pygame.display.set_caption(f"Training: {song_name}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)
            self.arrow_img = pygame.image.load("arrow.png").convert_alpha()
            self.arrow_img = pygame.transform.scale(self.arrow_img, (90, 90))

        self.episode_scores = []
        self.episode_stats = []
        self.epsilon_history = []

    def load_chart(self, filename):
        data = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts, direction = line.split(",")
                    t = float(ts.strip())
                    direction = direction.strip().lower()
                    data.append((t, direction))
                except ValueError:
                    continue
        print(f"✅ Chart cargado: {len(data)} notas desde {filename}")
        return data

    def _discretize_distance(self, dist):
        if dist == float('inf'):
            return "none"
        elif dist < 10:
            return "perfect"
        elif dist < 25:
            return "great"
        elif dist < 50:
            return "good"
        elif dist < 70:
            return "almost"
        else:
            return "far"

    def _state_from_arrows(self, arrows):
        # Min distancia por dirección
        dists = {"left": float('inf'), "down": float('inf'), "up": float('inf'), "right": float('inf')}
        for a in arrows:
            if not a.get("hit", False):
                d = abs(a["y"] - HIT_ZONE_Y)
                if d < dists[a["direction"]]:
                    dists[a["direction"]] = d
        return tuple(self._discretize_distance(dists[d]) for d in ["left","down","up","right"])

    def _exec_action(self, direction, arrows):
        # Busca la primera flecha de esa dirección no golpeada y la evalúa
        for a in arrows:
            if a["direction"] == direction and not a["hit"]:
                diff = abs(a["y"] - HIT_ZONE_Y)
                a["hit"] = True
                if diff < 10:
                    return "PERFECT", 100
                elif diff < 25:
                    return "GREAT", 50
                elif diff < 50:
                    return "GOOD", 20
                elif diff < 70:
                    return "ALMOST", 10
                else:
                    return "BOO", 0
        return "MISS", -10

    def train_episode(self, episode_num, total_episodes):
        arrows = []
        chart_index = 0
        score = 0
        stats = {"perfect":0,"great":0,"good":0,"almost":0,"boo":0,"miss":0}
        last_state = None
        last_action = None

        time_step = 1.0/60.0
        song_duration = (self.chart_data[-1][0] if self.chart_data else 0) + 2.0
        current_time = 0.0

        print(f"\n{'='*60}\n📀 Episodio {episode_num}/{total_episodes} - {self.song_name} | ε={self.agent.epsilon:.3f}\n{'='*60}")

        while current_time < song_duration:
            # Spawning
            while chart_index < len(self.chart_data) and current_time >= self.chart_data[chart_index][0]:
                _, direction = self.chart_data[chart_index]
                if self.visualize:
                    # Para visualizar, usamos objetos ligeros pero sin blit (mostramos círculos)
                    arrows.append({"direction": direction, "y": 720 + 50, "hit": False})
                else:
                    arrows.append({"direction": direction, "y": 720 + 50, "hit": False})
                chart_index += 1

            # Movimiento
            for a in arrows:
                a["y"] -= BASE_ARROW_SPEED

            # Eliminar salidas + contar misses
            before = len(arrows)
            arrows = [a for a in arrows if a["y"] > -50 and not a["hit"]]
            stats["miss"] += max(0, before - len(arrows))

            # Estado y acción
            state = self._state_from_arrows(arrows)
            action = self.agent.choose_action(state)

            if action != "none":
                judgement, delta = self._exec_action(action, arrows)
                if judgement and judgement != "MISS":
                    score += delta
                    stats[judgement.lower()] += 1
                    reward = self.agent.get_reward(judgement, delta)
                    next_state = self._state_from_arrows(arrows)
                    if last_state is not None and last_action is not None:
                        self.agent.update_q_value(last_state, last_action, reward, next_state)
                    last_state, last_action = state, action

            # Visual opcional mínima (no dibujamos assets para mantenerlo liviano)
            if self.visualize:
                # Eventos para cerrar
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return score, stats
                self.clock.tick(60)

            current_time += time_step

        total_notes = sum(stats.values())
        acc = (stats["perfect"] + stats["great"])/total_notes*100 if total_notes else 0.0
        print(f"🎯 Score: {score} | Accuracy: {acc:.2f}% | Notes: {total_notes}")
        return score, stats

    def train(self, episodes=50, epsilon_end=0.01, epsilon_decay_rate=None, save_interval=10):
        print(f"\n🎓 INICIANDO ENTRENAMIENTO | Canción: {self.song_name} | Episodios: {episodes}")
        # Cargar si existe
        self.agent.load_model(self.song_name)

        # Si no se especifica decay, calculamos geométrico para ir de start -> end en 'episodes'
        if epsilon_decay_rate is None:
            if self.agent.epsilon <= epsilon_end:
                epsilon_decay_rate = 1.0
            else:
                epsilon_decay_rate = (epsilon_end / self.agent.epsilon) ** (1.0 / max(1, episodes))

        start = time.time()
        for ep in range(1, episodes+1):
            score, stats = self.train_episode(ep, episodes)
            self.episode_scores.append(score)
            self.episode_stats.append(stats)
            self.epsilon_history.append(self.agent.epsilon)

            # actualizar métricas del agente
            self.agent.total_episodes += 1
            self.agent.training_scores.append(score)

            # epsilon decay
            self.agent.epsilon = max(epsilon_end, self.agent.epsilon * epsilon_decay_rate)

            if ep % save_interval == 0:
                self.agent.save_model(self.song_name)
                print(f"💾 Modelo guardado en episodio {ep}")

            if ep >= 10:
                avg10 = float(np.mean(self.episode_scores[-10:]))
                print(f"📈 Promedio últimos 10 episodios: {avg10:.1f}")

        self.agent.save_model(self.song_name)
        elapsed = time.time() - start
        print(f"\n✅ Entrenamiento completado en {elapsed/60:.1f} min. Modelo guardado.")

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento automático para agente DDR")
    parser.add_argument("--song", type=str, default="feel_the_power",
                        choices=["321stars","feel_the_power","lovelovesugar","mirror_dance"],
                        help="Canción a entrenar")
    parser.add_argument("--episodes", type=int, default=50, help="Número de episodios")
    parser.add_argument("--visualize", action="store_true", help="Muestra ventana mientras entrena (más lento)")
    parser.add_argument("--epsilon-start", type=float, default=0.5, help="Epsilon inicial")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Epsilon final")
    parser.add_argument("--alpha", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount")
    args = parser.parse_args()

    trainer = AutoTrainer(
        song_name=args.song,
        visualize=args.visualize,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
    )
    trainer.train(episodes=args.episodes, epsilon_end=args.epsilon_end)

if __name__ == "__main__":
    main()
