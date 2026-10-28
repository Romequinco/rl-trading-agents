"""Experimentos reproducibles del 03 sin modificar el entorno del taller.

La subclase solo amplía la observación. Ejecución, recompensa, comisión y balance
se heredan de SimpleTradingEnv. Los algoritmos reutilizan los agentes comunes.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import time

import gymnasium as gym
import numpy as np
import pandas as pd

from .agents import BuyHoldAgent, RandomAgent, QLearningAgent, SarsaAgent, HOLD
from .env import SimpleTradingEnv
from .experiment import (RESULTS_DIR, RESULT_COLUMNS, TRAIN_SEED_STRIDE,
                         VAL_SEED_OFFSET, TEST_SEED_OFFSET, episode_metrics, summarize)
from .markets import make_prices, register_extra_markets

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "sp500_2017_2025.csv"
CACHE = RESULTS_DIR / "cache" / "03_extras"
SYNTHETIC_MARKETS = ("reversion_media", "regimenes", "tendencia_persistente", "saltos")
LABELS = {
    "reversion_media": "Reversión a la media", "regimenes": "Regímenes",
    "tendencia_persistente": "Tendencia persistente", "saltos": "Difusión + saltos",
    "sp500_real": "S&P 500 · real",
}
ALGORITHMS = {"qlearning": QLearningAgent, "sarsa": SarsaAgent}
register_extra_markets()


@dataclass
class PreparedSeries:
    prices: np.ndarray
    features: np.ndarray
    dates: np.ndarray | None = None


def causal_features(prices):
    """[desviación MA6, desviación MA30, volatilidad de 20 retornos].

    Ventanas inclusivas en t: ninguna fila usa t+1. Los primeros pasos utilizan
    la historia disponible, también en el estado base. No hay centrado ni bfill.
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) < 7 or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Se requieren al menos 7 precios positivos y finitos")
    s = pd.Series(p)
    short = (s / s.rolling(6, min_periods=1).mean() - 1).clip(-1, 1)
    distance = (s / s.rolling(30, min_periods=1).mean() - 1).clip(-1, 1)
    vol = s.pct_change().rolling(20, min_periods=1).std(ddof=0).fillna(0)
    return np.column_stack((short, distance, vol)).astype(np.float32)


def prepare(prices, dates=None):
    prices = np.asarray(prices, dtype=float)
    return PreparedSeries(prices, causal_features(prices), dates)


class FeatureTradingEnv(SimpleTradingEnv):
    """Misma contabilidad del taller; observación causal de cuatro componentes."""
    def __init__(self, prepared):
        super().__init__(prepared.prices)
        self.features = prepared.features
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, np.inf, 1], dtype=np.float32), dtype=np.float32)

    def _get_obs(self):
        return np.array([*self.features[self.current_step], self.position], dtype=np.float32)


@dataclass
class StateEncoder:
    edges: tuple
    mode: str

    @property
    def n_states(self):
        return int(2 * np.prod([len(e) + 1 for e in self.edges]))

    def codes(self, features):
        """Índices pares; se suma después la posición 0/1."""
        out = np.zeros(len(features), dtype=np.int64)
        for j, edges in enumerate(self.edges):
            out = out * (len(edges) + 1) + np.digitize(features[:, j], edges)
        return 2 * out


def fit_encoder(series, mode="base"):
    if mode not in {"base", "enriquecido"}:
        raise ValueError(mode)
    x = np.concatenate([p.features[5:-1] for p in series])
    bins = (10,) if mode == "base" else (10, 3, 3)
    edges = tuple(np.unique(np.quantile(x[:, j], np.arange(1, b) / b))
                  for j, b in enumerate(bins))
    return StateEncoder(edges, mode)


class CashAgent(BuyHoldAgent):
    """Control de no operar: retorno y drawdown cero."""
    def act(self, state, greedy=False):
        return HOLD


def make_agent(cfg, encoder):
    agent = ALGORITHMS[cfg["algoritmo"]](
        edges=encoder.edges[0], alpha=cfg["alpha"], gamma=cfg["gamma"],
        epsilon_decay=cfg["epsilon_decay"], epsilon_min=cfg["epsilon_min"], seed=cfg["seed"])
    agent.Q = np.zeros((encoder.n_states, 3))
    return agent


def run_extra_episode(prepared, agent, encoder, learn=False, greedy=True, visits=None):
    env = FeatureTradingEnv(prepared)
    codes = encoder.codes(prepared.features)
    env.reset()
    s = int(codes[env.current_step] + env.position)
    a = agent.act(s, greedy)
    n = len(prepared.prices) - 6
    balances = np.empty(n + 1); balances[0] = env.balance
    actions = np.empty(n, dtype=int); positions = np.empty(n, dtype=int)
    total = 0.0
    for k in range(n):
        actions[k], positions[k] = a, env.position
        if visits is not None:
            visits[s] += 1
        _, r, done, truncated, _ = env.step(a)
        done = done or truncated
        s2 = int(codes[env.current_step] + env.position)
        if getattr(agent, "on_policy", False):
            a2 = agent.act(s2, greedy)
            if learn:
                agent.update(s, a, r, s2, a2, done)
        else:
            if learn:
                agent.update(s, a, r, s2, None, done)
            a2 = agent.act(s2, greedy)
        s, a = s2, a2
        total += r
        balances[k + 1] = env.balance
    return total, balances, actions, positions


@lru_cache(maxsize=1)
def real_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    meta = json.loads(DATA_FILE.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("El CSV real no coincide con su huella de procedencia")
    if not df.date.is_monotonic_increasing or not df.date.is_unique or df.close.isna().any():
        raise ValueError("Fechas o precios reales inválidos")
    return df, prepare(df.close.to_numpy(), df.date.to_numpy())


def real_indices(split):
    dates = real_data()[0].date
    masks = {"train": dates <= "2021-12-31",
             "val": (dates >= "2022-01-01") & (dates <= "2023-12-31"),
             "test": dates >= "2024-01-01"}
    return np.flatnonzero(masks[split])


def real_slice(start, stop):
    """Slice [start:stop], conservando indicadores calculados causalmente."""
    _, full = real_data()
    return PreparedSeries(full.prices[start:stop], full.features[start:stop], full.dates[start:stop])


def evaluation_series(market, split, n_steps=500, n_series=200):
    if market == "sp500_real":
        idx = real_indices(split)
        # Cinco cierres anteriores para empezar a operar en el primer cierre del split.
        return [real_slice(int(idx[0]) - 5, int(idx[-1]) + 1)]
    offset = {"val": VAL_SEED_OFFSET, "test": TEST_SEED_OFFSET}[split]
    return [prepare(make_prices(market, n_steps, offset + i)) for i in range(n_series)]


def calibration_series(market, n_steps=500):
    if market == "sp500_real":
        idx = real_indices("train")
        return [real_slice(int(idx[0]), int(idx[-1]) + 1)]
    # Las mismas 50 series de train para todos los algoritmos y semillas.
    return [prepare(make_prices(market, n_steps, i)) for i in range(50)]


def evaluate_prepared(agent, encoder, series, seed=0):
    old_rng = getattr(agent, "rng", None)
    if old_rng is not None:
        agent.rng = np.random.default_rng(200_000 + seed)
    rows = []
    try:
        for i, p in enumerate(series):
            _, balances, actions, positions = run_extra_episode(p, agent, encoder)
            row = {"serie": i, **episode_metrics(balances, actions, positions),
                   "exposicion": float(np.mean(np.where(actions == 2, 1, np.where(actions == 0, 0, positions))))}
            if p.dates is not None:
                row.update(fecha_inicio=str(pd.Timestamp(p.dates[5]).date()),
                           fecha_fin=str(pd.Timestamp(p.dates[-1]).date()))
            rows.append(row)
    finally:
        if old_rng is not None:
            agent.rng = old_rng
    return pd.DataFrame(rows)


def code_fingerprint():
    h = hashlib.sha256()
    for name in ("extras.py", "env.py", "agents.py", "markets.py", "experiment.py"):
        h.update((ROOT / "src" / name).read_bytes())
    h.update(DATA_FILE.read_bytes())
    return h.hexdigest()


def cache_id(cfg):
    payload = {**cfg, "code": code_fingerprint()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def train_config(cfg, recompute=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    cid = cache_id(cfg)
    js, npz = CACHE / f"{cid}.json", CACHE / f"{cid}.npz"
    if not recompute and js.exists() and npz.exists():
        return json.loads(js.read_text(encoding="utf-8"))
    encoder = fit_encoder(calibration_series(cfg["mercado"], cfg["n_steps"]), cfg["estado"])
    agent = make_agent(cfg, encoder)
    visits = np.zeros(encoder.n_states, dtype=np.int64)
    rewards = np.empty(cfg["n_episodes"])
    rng = np.random.default_rng(cfg["seed"])
    if cfg["mercado"] != "sp500_real":
        assert cfg["seed"] * TRAIN_SEED_STRIDE + cfg["n_episodes"] <= VAL_SEED_OFFSET
    t0 = time.perf_counter()
    for ep in range(cfg["n_episodes"]):
        if cfg["mercado"] == "sp500_real":
            idx = real_indices("train")
            start = int(rng.integers(idx[0], idx[-1] - cfg["real_window"] + 2))
            p = real_slice(start, start + cfg["real_window"])
        else:
            p = prepare(make_prices(cfg["mercado"], cfg["n_steps"], cfg["seed"] * TRAIN_SEED_STRIDE + ep))
        rewards[ep], *_ = run_extra_episode(p, agent, encoder, learn=True, greedy=False, visits=visits)
        agent.end_episode()
    elapsed = time.perf_counter() - t0
    val = evaluate_prepared(agent, encoder, evaluation_series(
        cfg["mercado"], "val", cfg["n_steps"], cfg["n_val"]), seed=cfg["seed"])
    row = {**cfg, **summarize(f"{cfg['algoritmo']}_{cfg['estado']}", cfg["mercado"], val, rewards, elapsed),
           "id": cid, "n_estados": encoder.n_states, "cobertura": float(np.mean(visits > 0)),
           "visitas_mediana": float(np.median(visits)), "split": "val"}
    np.savez_compressed(npz, Q=agent.Q, rewards=rewards, visits=visits,
                        **{f"edges{i}": e for i, e in enumerate(encoder.edges)})
    # NaN en std de la única trayectoria real se escribe como null en JSON estricto.
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
    js.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return clean


def load_model(cfg):
    row = train_config(cfg)
    with np.load(CACHE / f"{row['id']}.npz") as arr:
        n_features = 1 if cfg["estado"] == "base" else 3
        encoder = StateEncoder(tuple(arr[f"edges{i}"].copy() for i in range(n_features)), cfg["estado"])
        agent = make_agent(cfg, encoder); agent.Q[:] = arr["Q"]
        return agent, encoder, row, arr["rewards"].copy()


def test_config(cfg):
    agent, encoder, val_row, rewards = load_model(cfg)
    before = agent.Q.copy()
    df = evaluate_prepared(agent, encoder, evaluation_series(
        cfg["mercado"], "test", cfg["n_steps"], cfg["n_test"]), seed=cfg["seed"])
    assert np.array_equal(agent.Q, before), "El test no puede modificar Q"
    summary = {**val_row, **summarize(val_row["agente"], cfg["mercado"], df, rewards, val_row["tiempo_s"]), "split": "test"}
    for key in ("agente", "mercado", "algoritmo", "estado", "seed"):
        df[key] = summary[key]
    return summary, df.to_dict("records")


def _train_job(args):
    return train_config(*args)


def run_training(configs, name="03_validacion", n_jobs=6, recompute=False):
    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(_train_job, (cfg, recompute)) for cfg in configs]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 10 == 0 or i == len(configs):
                print(f"Entrenamiento/validación: {i}/{len(configs)} · {time.perf_counter()-start:.0f} s", flush=True)
    df = pd.DataFrame(rows).sort_values(["mercado", "algoritmo", "estado", "seed"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def run_test(configs, suffix="", n_jobs=6):
    summaries, episodes = [], []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i, (row, paths) in enumerate(pool.map(test_config, configs), 1):
            summaries.append(row); episodes.extend(paths)
            if i % 10 == 0 or i == len(configs):
                print(f"Test: {i}/{len(configs)}", flush=True)
    seeds = pd.DataFrame(summaries).sort_values(["mercado", "agente", "seed"])
    paths = pd.DataFrame(episodes)
    seeds.to_csv(RESULTS_DIR / f"03_extras_semillas{suffix}.csv", index=False)
    paths.to_csv(RESULTS_DIR / f"03_episodios_test{suffix}.csv", index=False)
    return seeds, paths


def baselines(markets, n_steps=500, n_test=200):
    rows, paths = [], []
    for market in markets:
        series = evaluation_series(market, "test", n_steps, n_test)
        encoder = fit_encoder(calibration_series(market, n_steps))
        for name, agent in (("aleatorio", RandomAgent(seed=0)), ("buy_hold", BuyHoldAgent()), ("no_operar", CashAgent())):
            df = evaluate_prepared(agent, encoder, series)
            rows.append(summarize(name, market, df))
            paths.extend(df.assign(agente=name, mercado=market, seed=-1).to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(paths)


def real_blocks(agent, encoder, block_size=63, seed=0):
    idx = real_indices("test")
    series = [real_slice(start - 5, min(start + block_size, int(idx[-1])) + 1)
              for start in range(int(idx[0]), int(idx[-1]), block_size)
              if min(start + block_size, int(idx[-1])) - start >= 21]
    return evaluate_prepared(agent, encoder, series, seed)


def paired_state_effect(paths, n_boot=2000):
    """IC bootstrap de mediana de deltas emparejados por serie, tras mediana entre semillas.

    Condicional a las políticas entrenadas: no es un IC sobre nuevos entrenamientos.
    """
    rng = np.random.default_rng(42)
    rows = []
    for market in SYNTHETIC_MARKETS:
        for algorithm in ALGORITHMS:
            sub = paths[(paths.mercado == market) & (paths.algoritmo == algorithm)]
            paired = sub.pivot_table(index="serie", columns="estado", values="retorno", aggfunc="median")
            delta = (paired.enriquecido - paired.base).to_numpy()
            boot = np.median(rng.choice(delta, (n_boot, len(delta)), replace=True), axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append(dict(mercado=market, algoritmo=algorithm, delta_mediana=np.median(delta),
                             ic95_inf=lo, ic95_sup=hi, fraccion_mejora=np.mean(delta > 1e-12)))
    return pd.DataFrame(rows)
