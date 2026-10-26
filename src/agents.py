"""Agentes con una interfaz común.

Todos exponen:
- `act(state, greedy=False) -> int`: acción para el estado discreto `state`.
- `update(s, a, r, s2, a2, done)`: aprendizaje tras un paso. `a2` es la acción que el agente
  ya ha elegido para `s2` (la usa SARSA; Q-Learning la ignora).
- `end_episode()`: se llama al final de cada episodio de entrenamiento (decaimiento de epsilon).

Los baselines no aprenden: `update` y `end_episode` no hacen nada.
"""

import numpy as np

SELL, HOLD, BUY = 0, 1, 2
N_ACTIONS = 3


class RandomAgent:
    """Línea base del taller: acción al azar en cada paso."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        return int(self.rng.integers(N_ACTIONS))

    def update(self, s, a, r, s2, a2, done):
        pass

    def end_episode(self):
        pass


class BuyHoldAgent:
    """Compra en el primer paso y mantiene hasta el final (comprar con posición abierta no hace nada)."""

    def act(self, state, greedy=False):
        return BUY

    def update(self, s, a, r, s2, a2, done):
        pass

    def end_episode(self):
        pass


class TabularAgent:
    """Base de los agentes tabulares: tabla Q y política epsilon-greedy con decaimiento."""

    def __init__(self, n_states, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, seed=None):
        self.Q = np.zeros((n_states, N_ACTIONS))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        q = self.Q[state]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # desempate aleatorio

    def update(self, s, a, r, s2, a2, done):
        raise NotImplementedError

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningAgent(TabularAgent):
    """Q-Learning (off-policy): objetivo r + gamma * max_a' Q(s2, a')."""

    def update(self, s, a, r, s2, a2, done):
        # TODO (01_qlearning)
        raise NotImplementedError


class SarsaAgent(TabularAgent):
    """SARSA (on-policy): objetivo r + gamma * Q(s2, a2)."""

    def update(self, s, a, r, s2, a2, done):
        # TODO (02_sarsa)
        raise NotImplementedError
