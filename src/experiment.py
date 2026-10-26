"""Protocolo común de entrenamiento y evaluación (D-0.1).

- Entrenamiento: cada episodio usa una serie nueva del mismo mercado. Con la semilla de entrenamiento `seed`,
  las series son `seed * TRAIN_SEED_STRIDE + 0, 1, 2, ...` (cada semilla ve series distintas).
- Validación (D-0.4): `N_VAL` series desde `VAL_SEED_OFFSET`. Solo se usan para elegir hiperparámetros.
- Test: `N_TEST` (200) series nuevas con semillas desde `TEST_SEED_OFFSET`. Solo para el resultado final.
- Resultados: `save_results()` escribe `results/NN_nombre.csv` con las columnas de `RESULT_COLUMNS`,
  para que el 04 solo tenga que concatenar.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .env import TREND_EDGES, SimpleTradingEnv, discretize
from .markets import make_prices

N_STEPS = 500
N_TEST = 200   # D-0.1: con 50, el error estándar de la mediana es de ±5-8 puntos (00 §5)
N_BINS = 10    # D-1.1: bins de tendencia por cuantiles de cada mercado (00 §7, recomendación del profesor)
TEST_SEED_OFFSET = 100_000
VAL_SEED_OFFSET = 50_000   # D-0.4
N_VAL = 100
TRAIN_SEED_STRIDE = 10_000
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULT_COLUMNS = ["agente", "mercado", "retorno_mediana", "retorno_medio", "std", "sharpe", "max_drawdown",
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


def evaluate(agent, market, n_test=None, n_steps=N_STEPS, greedy=True, split="test"):
    """Evalúa sin aprender sobre series de `split` ("test" o "val"). Devuelve un DataFrame (una fila por serie)."""
    offset = {"test": TEST_SEED_OFFSET, "val": VAL_SEED_OFFSET}[split]
    n_test = n_test or {"test": N_TEST, "val": N_VAL}[split]
    rows = []
    for i in range(n_test):
        env = SimpleTradingEnv(make_prices(market, n_steps, offset + i))
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


def quantile_edges(market, n_bins=N_BINS, n_series=50, n_steps=N_STEPS, seed_offset=0):
    """Bordes de tendencia por cuantiles de `market`, calculados solo con series de train (D-1.1)."""
    return np.array(_quantile_edges(market, n_bins, n_series, n_steps, seed_offset))


@lru_cache(maxsize=None)
def _quantile_edges(market, n_bins, n_series, n_steps, seed_offset):
    trends = []
    for i in range(n_series):
        w = np.lib.stride_tricks.sliding_window_view(make_prices(market, n_steps, seed_offset + i), 6)
        trends.append((w[:, -1] - w.mean(axis=1)) / w.mean(axis=1))
    return tuple(np.quantile(np.clip(np.concatenate(trends), -1.0, 1.0), np.linspace(0, 1, n_bins + 1)[1:-1]))


def summarize(agent_name, market, test_df, rewards=None, train_time=0.0):
    """Fila resumen con el esquema común `RESULT_COLUMNS`."""
    return {
        "agente": agent_name,
        "mercado": market,
        "retorno_mediana": test_df["retorno"].median(),
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


# ---------------------------------------------------------------------------
# Barridos de hiperparámetros en paralelo, con caché (01, 02)
# ---------------------------------------------------------------------------
CONFIG_KEYS = ["agente", "mercado", "n_bins", "alpha", "gamma", "epsilon_decay", "epsilon_min",
               "n_episodes", "seed", "split"]
CACHE_DIR = RESULTS_DIR / "cache"   # curvas y tablas Q (no versionado; se regenera)


def config_id(cfg):
    """Identificador estable de una configuración (los números se normalizan a float para que coincidan al leer el CSV)."""
    norm = lambda v: float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v
    return "|".join(f"{k}={norm(cfg[k])}" for k in CONFIG_KEYS)


def run_config(cfg):
    """Entrena y evalúa una configuración (dict con `CONFIG_KEYS`). Devuelve (fila, curva, tabla Q)."""
    from .agents import QLearningAgent, SarsaAgent
    cls = {"qlearning": QLearningAgent, "sarsa": SarsaAgent}[cfg["agente"]]
    offset = cfg["seed"] * TRAIN_SEED_STRIDE
    assert offset + cfg["n_episodes"] <= VAL_SEED_OFFSET, "las series de train invadirían las de validación"
    agent = cls(edges=quantile_edges(cfg["mercado"], cfg["n_bins"]), alpha=cfg["alpha"], gamma=cfg["gamma"],
                epsilon_decay=cfg["epsilon_decay"], epsilon_min=cfg["epsilon_min"], seed=cfg["seed"])
    rewards, t = train(agent, cfg["mercado"], cfg["n_episodes"], seed_offset=offset)
    res = summarize(cfg["agente"], cfg["mercado"], evaluate(agent, cfg["mercado"], split=cfg["split"]), rewards, t)
    return {**{k: cfg[k] for k in CONFIG_KEYS}, **{k: v for k, v in res.items() if k not in ("agente", "mercado")}},         rewards, agent.Q


def run_grid(configs, name, n_jobs=None, recompute=False):
    """Ejecuta `configs` en paralelo y cachea los resultados.

    Resúmenes en `results/<name>.csv` (versionado) y curvas + tablas Q en `results/cache/<name>.npz`.
    Solo se entrenan las configuraciones que faltan. Devuelve (DataFrame, {id: curva}, {id: Q}).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    csv, npz = RESULTS_DIR / f"{name}.csv", CACHE_DIR / f"{name}.npz"
    rows, curves, qs = {}, {}, {}
    if not recompute and csv.exists() and npz.exists():
        old = pd.read_csv(csv)
        arrays = np.load(npz)
        for r in old.to_dict("records"):
            cid = config_id(r)
            if f"r::{cid}" in arrays:
                rows[cid], curves[cid], qs[cid] = r, arrays[f"r::{cid}"], arrays[f"q::{cid}"]
    todo = [c for c in configs if config_id(c) not in rows]
    if todo:
        n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
        print(f"Entrenando {len(todo)} configuraciones en {n_jobs} procesos "
              f"({len(configs) - len(todo)} ya estaban en caché)...")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for cfg, (row, rew, q) in zip(todo, pool.map(run_config, todo)):
                cid = config_id(cfg)
                rows[cid], curves[cid], qs[cid] = row, rew, q
        print(f"Hecho en {time.perf_counter() - t0:.0f} s")
        pd.DataFrame(list(rows.values())).to_csv(csv, index=False)
        np.savez_compressed(npz, **{f"r::{k}": v for k, v in curves.items()}, **{f"q::{k}": v for k, v in qs.items()})
    ids = [config_id(c) for c in configs]
    return (pd.DataFrame([rows[i] for i in ids]).assign(id=ids),
            {i: curves[i] for i in ids}, {i: qs[i] for i in ids})


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
