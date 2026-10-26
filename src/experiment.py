"""Protocolo común de entrenamiento y evaluación (D-0.1).

- Entrenamiento: cada episodio usa una serie nueva del mismo mercado (semillas 0, 1, 2, ...).
- Test: `N_TEST` series nuevas con semillas desde `TEST_SEED_OFFSET`, que nunca se ven al entrenar.
- Resultados: `save_results()` escribe `results/NN_nombre.csv` con las columnas de `RESULT_COLUMNS`,
  para que el 04 solo tenga que concatenar.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .env import TREND_EDGES, SimpleTradingEnv, discretize
from .markets import make_prices

N_STEPS = 500
N_TEST = 50
TEST_SEED_OFFSET = 100_000
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULT_COLUMNS = ["agente", "mercado", "retorno_medio", "std", "sharpe", "max_drawdown",
                  "n_trades", "episodios_convergencia", "tiempo_s"]


def run_episode(env, agent, learn=True, greedy=False):
    """Juega un episodio completo. Devuelve recompensa total, balances, acciones y posiciones.

    Sigue el orden de los algoritmos de slides3: SARSA elige A' antes de actualizar (p31) y
    Q-Learning actualiza y luego elige la acción con la Q ya actualizada (p36).
    """
    edges = getattr(agent, "edges", TREND_EDGES)
    on_policy = getattr(agent, "on_policy", False)
    obs, _ = env.reset()
    s = discretize(obs, edges)
    a = agent.act(s, greedy)
    total_reward = 0.0
    balances, actions, positions = [env.balance], [], []
    terminated = False
    while not terminated:
        positions.append(env.position)
        actions.append(a)
        obs, r, terminated, truncated, info = env.step(a)
        s2 = discretize(obs, edges)
        if on_policy:
            a2 = agent.act(s2, greedy)
            if learn:
                agent.update(s, a, r, s2, a2, terminated)
        else:
            if learn:
                agent.update(s, a, r, s2, None, terminated)
            a2 = agent.act(s2, greedy)
        s, a = s2, a2
        total_reward += r
        balances.append(env.balance)
    return total_reward, np.array(balances), np.array(actions), np.array(positions)


def train(agent, market, n_episodes, n_steps=N_STEPS, seed_offset=0):
    """Entrena `agent` con `n_episodes` series distintas. Devuelve la curva de recompensa y el tiempo."""
    rewards = np.zeros(n_episodes)
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        env = SimpleTradingEnv(make_prices(market, n_steps, seed_offset + ep))
        rewards[ep], *_ = run_episode(env, agent, learn=True)
        agent.end_episode()
    return rewards, time.perf_counter() - t0


def episode_metrics(balances, actions, positions):
    """Métricas de un episodio a partir de la evolución del balance (Sharpe por paso, sin anualizar)."""
    step_returns = np.diff(balances) / balances[:-1]
    peak = np.maximum.accumulate(balances)
    # Operación efectiva: comprar sin posición o vender con posición
    trades = int(np.sum(((actions == 2) & (positions == 0)) | ((actions == 0) & (positions == 1))))
    return {
        "retorno": balances[-1] / balances[0] - 1,
        "sharpe": step_returns.mean() / step_returns.std() if step_returns.std() > 0 else 0.0,
        "max_drawdown": float(np.max(1 - balances / peak)),
        "n_trades": trades,
    }


def evaluate(agent, market, n_test=N_TEST, n_steps=N_STEPS, greedy=True):
    """Evalúa sin aprender sobre `n_test` series de test. Devuelve un DataFrame (una fila por serie)."""
    rows = []
    for i in range(n_test):
        env = SimpleTradingEnv(make_prices(market, n_steps, TEST_SEED_OFFSET + i))
        _, balances, actions, positions = run_episode(env, agent, learn=False, greedy=greedy)
        rows.append(episode_metrics(balances, actions, positions))
    return pd.DataFrame(rows)


def convergence_episode(rewards, window=50, frac=0.9):
    """Primer episodio en que la media móvil recorre `frac` del camino entre su valor inicial y el final
    (episodios hasta el 90 % de la mejora). Robusto al ruido de la curva de recompensa."""
    if len(rewards) < window:
        return len(rewards)
    ma = pd.Series(rewards).rolling(window).mean().to_numpy()[window - 1:]
    target = ma[0] + frac * (ma[-1] - ma[0])
    reached = ma >= target if ma[-1] >= ma[0] else ma <= target
    return int(np.argmax(reached) + window - 1)


def quantile_edges(market, n_bins=5, n_series=50, n_steps=N_STEPS, seed_offset=0):
    """Bordes de tendencia por cuantiles de `market`, calculados solo con series de train (D-1.1)."""
    trends = []
    for i in range(n_series):
        prices = make_prices(market, n_steps, seed_offset + i)
        for t in range(5, n_steps):
            window = prices[t - 5 : t + 1]
            trends.append((window[-1] - np.mean(window)) / np.mean(window))
    return np.quantile(np.clip(trends, -1.0, 1.0), np.linspace(0, 1, n_bins + 1)[1:-1])


def summarize(agent_name, market, test_df, rewards=None, train_time=0.0):
    """Fila resumen con el esquema común `RESULT_COLUMNS`."""
    return {
        "agente": agent_name,
        "mercado": market,
        "retorno_medio": test_df["retorno"].mean(),
        "std": test_df["retorno"].std(),
        "sharpe": test_df["sharpe"].mean(),
        "max_drawdown": test_df["max_drawdown"].mean(),
        "n_trades": test_df["n_trades"].mean(),
        "episodios_convergencia": convergence_episode(rewards) if rewards is not None else np.nan,
        "tiempo_s": train_time,
    }


def save_results(rows, name):
    """Guarda las filas resumen en `results/<name>.csv` (p. ej. name='01_qlearning')."""
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def plot_episode(prices, balances, positions, label="Agente"):
    """Figura de dos paneles del taller: precio con posición (verde/rojo) y balance frente a buy & hold."""
    valid_prices = prices[5:-1]
    on = np.where(positions == 1, valid_prices, np.nan)
    off = np.where(positions == 0, valid_prices, np.nan)
    bh = 1000.0 * 0.999 / prices[5] * prices[5:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(valid_prices, color="black", alpha=0.7, linewidth=1.5, label="Precio")
    ax1.plot(on, "g.", label="Con activo")
    ax1.plot(off, "r.", label="Sin activo")
    ax1.set_ylabel("Precio ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2.plot(balances, color="blue", linewidth=2, label=label)
    ax2.plot(bh, color="orange", linestyle="--", linewidth=2, label="Buy & Hold")
    ax2.axhline(1000, color="gray", linestyle=":", label="Capital inicial")
    ax2.set_xlabel("Pasos de tiempo")
    ax2.set_ylabel("Balance ($)")
    ax2.legend(loc="upper left")
    ax2.grid(True)
    fig.tight_layout()
    return fig


def plot_learning_curve(rewards, window=50, label=None, ax=None):
    """Recompensa por episodio de entrenamiento y su media móvil."""
    ax = ax or plt.gca()
    ax.plot(rewards, alpha=0.25)
    ax.plot(pd.Series(rewards).rolling(window).mean(), label=label)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa total")
    ax.grid(True)
    return ax
