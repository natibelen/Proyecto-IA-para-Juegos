import numpy as np
import pickle
import os
from collections import defaultdict

class DDRAgent:
    """
    Agente inteligente para Dance Dance Revolution que aprende mediante Q-Learning.
    
    TEORÍA:
    -------
    Q-Learning es un algoritmo de aprendizaje por refuerzo que aprende una función Q(estado, acción)
    que estima la recompensa futura esperada de tomar una acción en un estado dado.
    
    La ecuación de actualización de Q-Learning es:
    Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    
    Donde:
    - α (alpha) = tasa de aprendizaje (cuánto actualizamos en cada paso)
    - γ (gamma) = factor de descuento (cuánto valoramos recompensas futuras)
    - r = recompensa inmediata
    - s' = nuevo estado después de tomar la acción
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        Parámetros:
        -----------
        alpha: tasa de aprendizaje (0-1). Qué tan rápido aprende de nuevas experiencias
        gamma: factor de descuento (0-1). Importancia de recompensas futuras vs inmediatas
        epsilon: tasa de exploración (0-1). Probabilidad de tomar acciones aleatorias
        """
        self.alpha = alpha      # Tasa de aprendizaje
        self.gamma = gamma      # Factor de descuento
        self.epsilon = epsilon  # Exploración vs Explotación
        
        # Tabla Q: almacena Q(estado, acción) para cada par estado-acción
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Estadísticas de entrenamiento
        self.training_scores = []
        self.total_episodes = 0
        
    def get_state(self, arrows, hit_zone_y=100):
        """
        Extrae características del estado actual del juego.
        
        ESTADO: Representación discreta de la situación actual
        -------
        El estado incluye información sobre las flechas más cercanas en cada dirección:
        - Distancia a la zona de golpe
        - Si hay flecha en esa dirección
        - Qué tan cerca están múltiples flechas
        
        Esto es similar a cómo en Pacman observábamos posición, comida cercana y fantasmas.
        """
        state_features = {
            "left": float('inf'),
            "down": float('inf'),
            "up": float('inf'),
            "right": float('inf')
        }
        
        # Encontrar la flecha más cercana en cada dirección
        for arrow in arrows:
            if not arrow.hit:
                distance = abs(arrow.y - hit_zone_y)
                direction = arrow.direction
                if distance < state_features[direction]:
                    state_features[direction] = distance
        
        # Discretizar distancias para reducir espacio de estados
        # Similar a agrupar posiciones en Pacman
        def discretize_distance(dist):
            if dist == float('inf'):
                return "none"
            elif dist < 10:
                return "perfect"  # Momento perfecto
            elif dist < 25:
                return "great"    # Momento bueno
            elif dist < 50:
                return "good"     # Momento aceptable
            elif dist < 70:
                return "almost"   # Momento casi
            else:
                return "far"      # Muy lejos
        
        # Crear tupla de estado (debe ser hasheable para usarla como clave)
        state = tuple(discretize_distance(state_features[d]) 
                     for d in ["left", "down", "up", "right"])
        
        return state
    
    def choose_action(self, state, available_actions=["left", "down", "up", "right"]):
        """
        Selecciona una acción usando estrategia ε-greedy.
        
        EXPLORACIÓN vs EXPLOTACIÓN:
        ----------------------------
        - Con probabilidad ε: exploramos (acción aleatoria) para descubrir mejores estrategias
        - Con probabilidad 1-ε: explotamos (mejor acción conocida) para maximizar recompensa
        
        Esto es crucial porque:
        - Mucha exploración → el agente prueba cosas pero no usa lo que aprende
        - Mucha explotación → el agente se queda atascado en estrategias subóptimas
        """
        # Exploración: acción aleatoria
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        
        # Explotación: mejor acción según Q-table
        q_values = {action: self.q_table[state][action] for action in available_actions}
        max_q = max(q_values.values())
        
        # Si hay empate, elegir aleatoriamente entre las mejores
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)
    
    def evaluation_function(self, state, action, arrows, hit_zone_y=100):
        """
        Función de evaluación heurística (similar a tu código de Pacman).
        
        HEURÍSTICA:
        -----------
        Proporciona una evaluación inicial antes de que el agente aprenda mucho.
        Similar a cómo en Pacman evaluabas distancia a comida, fantasmas, etc.
        
        Esta función "enseña" al agente qué es importante:
        - Presionar cuando la flecha está cerca (reward positivo)
        - NO presionar cuando no hay flechas (penalización)
        - Bonus por timing perfecto
        """
        score = 0.0
        
        # Encontrar flecha más cercana en la dirección de la acción
        closest_arrow_distance = float('inf')
        for arrow in arrows:
            if arrow.direction == action and not arrow.hit:
                distance = abs(arrow.y - hit_zone_y)
                if distance < closest_arrow_distance:
                    closest_arrow_distance = distance
        
        # Recompensar presionar cuando hay flecha cerca
        if closest_arrow_distance != float('inf'):
            if closest_arrow_distance < 10:
                score += 100.0  # PERFECT timing
            elif closest_arrow_distance < 25:
                score += 50.0   # GREAT timing
            elif closest_arrow_distance < 50:
                score += 20.0   # GOOD timing
            elif closest_arrow_distance < 70:
                score += 10.0   # ALMOST
            else:
                score -= 5.0    # Demasiado temprano/tarde
        else:
            # Penalizar presionar cuando no hay flecha
            score -= 15.0
        
        # Penalizar no hacer nada cuando hay flechas cerca
        any_close_arrow = any(abs(arrow.y - hit_zone_y) < 50 and not arrow.hit 
                             for arrow in arrows)
        if action == "none" and any_close_arrow:
            score -= 20.0
        
        return score
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Actualiza la tabla Q usando la ecuación de Q-Learning.
        
        ACTUALIZACIÓN Q-LEARNING:
        -------------------------
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Intuición:
        - Si la recompensa real (r + γ·max(Q(s',a'))) es mayor que nuestra estimación Q(s,a),
          incrementamos Q(s,a) para acercarnos a la realidad
        - Si es menor, decrementamos Q(s,a)
        - α controla qué tan rápido hacemos estos ajustes
        """
        # Q actual para este par estado-acción
        current_q = self.q_table[state][action]
        
        # Mejor Q-value posible del siguiente estado
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        # Ecuación de Q-Learning
        # TD Error (Temporal Difference) = r + γ·max(Q(s',a')) - Q(s,a)
        td_error = reward + self.gamma * max_next_q - current_q
        
        # Actualización: Q(s,a) ← Q(s,a) + α·TD_error
        self.q_table[state][action] = current_q + self.alpha * td_error
    
    def get_reward(self, judgement, score_gained, missed=False):
        """
        Calcula la recompensa basada en el resultado de la acción.
        
        FUNCIÓN DE RECOMPENSA:
        ----------------------
        Define qué es "bueno" y "malo" para el agente.
        Similar a cómo en Pacman ganar daba +inf y perder -inf.
        
        Recompensas bien diseñadas son cruciales para el aprendizaje.
        """
        if missed:
            return -30.0  # Perder una nota es muy malo
        
        reward_map = {
            "PERFECT": 100.0,
            "GREAT": 50.0,
            "GOOD": 20.0,
            "ALMOST": 10.0,
            "BOO": -10.0
        }
        
        return reward_map.get(judgement, 0.0) + score_gained * 0.1
    
    def train_episode(self, game_instance):
        """
        Entrena el agente jugando un episodio completo.
        
        EPISODIO:
        ---------
        Una ejecución completa del juego desde inicio hasta fin.
        Similar a jugar una partida completa de Pacman.
        
        Durante el entrenamiento:
        1. Observamos el estado
        2. Elegimos una acción
        3. Ejecutamos la acción
        4. Observamos recompensa y nuevo estado
        5. Actualizamos Q-table
        6. Repetimos
        """
        episode_score = 0
        actions_taken = 0
        
        # Aquí se jugaría el juego y se actualizarían los Q-values
        # Este es un esqueleto - necesitarías integrarlo con tu game loop
        
        self.total_episodes += 1
        self.training_scores.append(episode_score)
        
        # Decaimiento de epsilon (exploración)
        # Con el tiempo, exploramos menos y explotamos más
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return episode_score
    
    def save_model(self, filename="ddr_agent.pkl"):
        """Guarda el modelo entrenado"""
        with open(filename, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "total_episodes": self.total_episodes,
                "training_scores": self.training_scores
            }, f)
        print(f"✅ Modelo guardado en {filename}")
    
    def load_model(self, filename="ddr_agent.pkl"):
        """Carga un modelo previamente entrenado"""
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), data["q_table"])
                self.alpha = data["alpha"]
                self.gamma = data["gamma"]
                self.epsilon = data["epsilon"]
                self.total_episodes = data["total_episodes"]
                self.training_scores = data["training_scores"]
            print(f"✅ Modelo cargado desde {filename}")
            return True
        return False


# ============================================================================
# INTEGRACIÓN CON TU JUEGO
# ============================================================================

class DDRAgentIntegration:
    """
    Clase para integrar el agente con tu RhythmGame.
    
    USO:
    ----
    1. Crear agente
    2. Entrenar durante varios episodios
    3. Usar modo de juego del agente
    """
    
    def __init__(self, game_instance):
        self.game = game_instance
        self.agent = DDRAgent(alpha=0.1, gamma=0.9, epsilon=0.3)
        self.last_state = None
        self.last_action = None
        
    def agent_play_step(self, arrows, hit_zone_y=100):
        """
        Un paso de juego del agente (llamar en cada frame del game loop).
        
        INTEGRACIÓN:
        ------------
        Esta función se llama desde tu game_loop() en cada frame.
        El agente observa, decide y actúa.
        """
        # 1. OBSERVAR estado actual
        current_state = self.agent.get_state(arrows, hit_zone_y)
        
        # 2. ELEGIR acción (incluye "none" para no presionar)
        action = self.agent.choose_action(current_state, 
                                         ["left", "down", "up", "right", "none"])
        
        # 3. EJECUTAR acción en el juego
        if action != "none":
            # Verificar resultado
            judgement, score_gained = self.execute_action(action, arrows, hit_zone_y)
            
            # 4. CALCULAR recompensa
            reward = self.agent.get_reward(judgement, score_gained)
            
            # 5. OBSERVAR nuevo estado
            next_state = self.agent.get_state(arrows, hit_zone_y)
            
            # 6. APRENDER (actualizar Q-table)
            if self.last_state is not None:
                self.agent.update_q_value(self.last_state, self.last_action, 
                                         reward, current_state)
            
            # 7. Preparar para siguiente paso
            self.last_state = current_state
            self.last_action = action
            
            return action, judgement, score_gained
        
        return None, None, 0
    
    def execute_action(self, direction, arrows, hit_zone_y):
        """
        Ejecuta la acción del agente y retorna el resultado.
        Similar a tu handle_input() pero retorna información para aprendizaje.
        """
        for arrow in arrows:
            if arrow.direction == direction and not arrow.hit:
                diff = abs(arrow.y - hit_zone_y)
                if diff < 10:
                    arrow.hit = True
                    return "PERFECT", 100
                elif diff < 25:
                    arrow.hit = True
                    return "GREAT", 50
                elif diff < 50:
                    arrow.hit = True
                    return "GOOD", 20
                elif diff < 70:
                    arrow.hit = True
                    return "ALMOST", 10
                else:
                    arrow.hit = True
                    return "BOO", 0
        
        return "MISS", -10


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

"""
CÓMO USAR EL AGENTE EN TU JUEGO:
---------------------------------

1. En tu clase RhythmGame.__init__():
   self.agent_integration = DDRAgentIntegration(self)
   self.agent_mode = False  # Variable para activar/desactivar agente

2. En tu game_loop(), después de handle_input pero antes de update:
   
   if self.agent_mode:
       action, judgement, score = self.agent_integration.agent_play_step(
           self.arrows, HIT_ZONE_Y
       )
       if action:
           self.score += score
           self.judgement = judgement

3. Para entrenar:
   - Activar modo agente
   - Jugar varias canciones (episodios)
   - El agente aprenderá qué acciones dar mejores resultados
   - Guardar modelo: agent_integration.agent.save_model()

4. Para usar agente entrenado:
   - Cargar modelo: agent_integration.agent.load_model()
   - Activar modo agente
   - El agente jugará usando lo aprendido

ESTRATEGIA DE ENTRENAMIENTO:
-----------------------------
1. Fase 1 (Exploración): ε = 0.3-0.5, aprender patrones básicos
2. Fase 2 (Refinamiento): ε = 0.1-0.2, mejorar timing
3. Fase 3 (Explotación): ε = 0.01-0.05, jugar óptimamente

MÉTRICAS DE PROGRESO:
---------------------
- Score promedio por episodio (debe aumentar)
- Porcentaje de PERFECTs (debe aumentar)
- Porcentaje de MISSes (debe disminuir)
"""