"""Anexo del 03: pruebas de robustez con el mismo protocolo.

Reutiliza series, indicadores causales, codificación del estado y métricas de `extras.py`,
que no se modifica. Solo añade:
- subconjuntos del estado enriquecido (solo MA30, solo volatilidad);
- comisión por operación configurable (`CostTradingEnv`; con 0,1 % reproduce el taller);
- recompensa logarítmica del balance;
- Expected SARSA y Double Q-Learning;
- la serie real con retornos barajados (placebo) y cortes anuales (walk-forward).
Las configuraciones se fijan antes del test: el anexo no selecciona nada mirando resultados.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .agents import TabularAgent, QLearningAgent, SarsaAgent, BuyHoldAgent, N_ACTIONS
from .experiment import RESULTS_DIR, TRAIN_SEED_STRIDE, VAL_SEED_OFFSET, episode_metrics, summarize
from .markets import make_prices
from .extras import (ROOT, PreparedSeries, FeatureTradingEnv, StateEncoder, CashAgent, prepare,
                     calibration_series, evaluation_series, real_data, real_indices,
                     code_fingerprint as extras_fingerprint)

CACHE = RESULTS_DIR / "cache" / "03_anexo"
# Columnas de `causal_features`: 0 tendencia MA6, 1 desviación MA30, 2 volatilidad20.
COLUMNS = {"base": (0,), "ma30": (0, 1), "vol": (0, 2), "enriquecido": (0, 1, 2)}
BINS = (10, 3, 3)
STATE_LABELS = {"base": "Base", "ma30": "Base + MA30", "vol": "Base + volatilidad",
                "enriquecido": "Base + MA30 + volatilidad"}
ALGORITHM_LABELS = {"qlearning": "Q-Learning", "sarsa": "SARSA",
                    "expected_sarsa": "Expected SARSA", "double_q": "Double Q-Learning"}


# ---------------------------------------------------------------------------
# Estado: cualquier subconjunto de los indicadores del 03
# ---------------------------------------------------------------------------
@dataclass
class SubsetEncoder(StateEncoder):
    cols: tuple = (0,)

    def codes(self, features):
        return super().codes(features[:, list(self.cols)])


def fit_subset_encoder(series, mode):
    """Igual que `fit_encoder` (cuantiles solo de train), para cualquier subconjunto de columnas."""
    cols = COLUMNS[mode]
    x = np.concatenate([p.features[5:-1] for p in series])
    edges = tuple(np.unique(np.quantile(x[:, j], np.arange(1, BINS[j]) / BINS[j])) for j in cols)
    return SubsetEncoder(edges, mode, cols)


# ---------------------------------------------------------------------------
# Entorno con comisión configurable
# ---------------------------------------------------------------------------
class CostTradingEnv(FeatureTradingEnv):
    """Ejecución y contabilidad del taller con la comisión como parámetro (0,1 % por defecto)."""
    def __init__(self, prepared, cost_rate=0.001):
        super().__init__(prepared)
        self.cost_rate = cost_rate

    def step(self, action):
        # Copia de SimpleTradingEnv.step; solo cambia cost_rate (una prueba lo contrasta con el original).
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]
        reward = 0.0
        if action == 2 and self.position == 0:
            effective_cash = self.cash - self.cash * self.cost_rate
            self.shares = effective_cash / prev_price
            self.cash = 0.0
            self.position = 1
            reward -= self.cost_rate
        elif action == 0 and self.position == 1:
            sale_value = self.shares * prev_price
            self.cash = sale_value - sale_value * self.cost_rate
            self.shares = 0.0
            self.position = 0
            reward -= self.cost_rate
        if self.position == 1:
            reward += (curr_price - prev_price) / prev_price
            self.balance = self.cash + self.shares * curr_price
        else:
            self.balance = self.cash
        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {"balance": self.balance}


# ---------------------------------------------------------------------------
# Dos algoritmos TD más, con la interfaz común
# ---------------------------------------------------------------------------
class ExpectedSarsaAgent(TabularAgent):
    """Expected SARSA: objetivo r + gamma * sum_a' pi(a'|S') Q(S', a'), con la pi epsilon-greedy vigente.

    No necesita A': se actualiza y después se elige la acción, como en Q-Learning.
    """

    def update(self, s, a, r, s2, a2, done):
        if done:
            target = r
        else:
            q = self.Q[s2]
            best = q == q.max()
            pi = np.full(N_ACTIONS, self.epsilon / N_ACTIONS) + best * (1 - self.epsilon) / best.sum()
            target = r + self.gamma * pi @ q
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class DoubleQLearningAgent(TabularAgent):
    """Double Q-Learning: una tabla elige la acción de S' y la otra la valora; actúa con QA + QB.

    Separar elección y valoración corrige el sesgo optimista de max_a Q con estimaciones ruidosas.
    """

    @property
    def Q(self):
        return self.QA + self.QB

    @Q.setter
    def Q(self, value):
        self.QA = np.array(value, dtype=float)
        self.QB = self.QA.copy()

    def update(self, s, a, r, s2, a2, done):
        A, B = (self.QA, self.QB) if self.rng.random() < 0.5 else (self.QB, self.QA)
        if done:
            target = r
        else:
            q = A[s2]
            target = r + self.gamma * B[s2, int(self.rng.choice(np.flatnonzero(q == q.max())))]
        A[s, a] += self.alpha * (target - A[s, a])


ALGORITHMS = {"qlearning": QLearningAgent, "sarsa": SarsaAgent,
              "expected_sarsa": ExpectedSarsaAgent, "double_q": DoubleQLearningAgent}


def make_agent(cfg, encoder):
    agent = ALGORITHMS[cfg["algoritmo"]](
        edges=encoder.edges[0], alpha=cfg["alpha"], gamma=cfg["gamma"],
        epsilon_decay=cfg["epsilon_decay"], epsilon_min=cfg["epsilon_min"], seed=cfg["seed"])
    agent.Q = np.zeros((encoder.n_states, N_ACTIONS))
    return agent


def run_episode_anexo(prepared, agent, encoder, cost=0.001, reward="taller",
                      learn=False, greedy=True, visits=None):
    """`run_extra_episode` con comisión configurable y recompensa opcional log(balance_t / balance_t-1)."""
    env = CostTradingEnv(prepared, cost)
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
        before = env.balance
        _, r, done, truncated, _ = env.step(a)
        if reward == "log":
            r = float(np.log(env.balance / before))
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


# ---------------------------------------------------------------------------
# Fuentes de datos: sintéticos, S&P 500, S&P 500 barajado y cortes anuales
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def shuffled_real(perm_seed):
    """S&P 500 con los log-retornos permutados dentro de cada split (placebo).

    Conserva el cierre en cada frontera de split, así que buy & hold obtiene exactamente el mismo
    retorno en test que en la serie real; solo se destruye la dependencia temporal.
    """
    _, full = real_data()
    logp = np.log(full.prices)
    r = np.diff(logp)  # r[k]: retorno del cierre k al k+1
    rng = np.random.default_rng(perm_seed)
    for split in ("train", "val", "test"):
        k = real_indices(split)
        k = k[k < len(r)]
        r[k] = rng.permutation(r[k])
    return prepare(np.exp(np.r_[logp[0], logp[0] + np.cumsum(r)]), full.dates)


def _cut(full, start, stop):
    return PreparedSeries(full.prices[start:stop], full.features[start:stop], full.dates[start:stop])


def real_ranges(cfg):
    """Índices de train y test. En el corte anual se entrena con todo lo anterior al año de test."""
    dates = real_data()[0].date
    if cfg["mercado"] == "sp500_anual":
        year = cfg["anio"]
        return np.flatnonzero(dates < f"{year}-01-01"), np.flatnonzero(dates.dt.year == year)
    return real_indices("train"), real_indices("test")


@lru_cache(maxsize=32)
def _synthetic(market, split, n_steps, n_test):
    if split == "calibracion":
        return tuple(calibration_series(market, n_steps))
    return tuple(evaluation_series(market, "test", n_steps, n_test))


def sources(cfg):
    """(series de calibración de bins, series de test, función que devuelve la serie del episodio)."""
    if cfg["mercado"].startswith("sp500"):
        full = shuffled_real(cfg["seed"]) if cfg["mercado"] == "sp500_barajado" else real_data()[1]
        train, test = real_ranges(cfg)
        window = cfg["real_window"]

        def episode(ep, rng):
            start = int(rng.integers(train[0], train[-1] - window + 2))
            return _cut(full, start, start + window)
        # Cinco cierres anteriores para empezar a operar en el primer cierre del test.
        return [_cut(full, train[0], train[-1] + 1)], [_cut(full, test[0] - 5, test[-1] + 1)], episode

    assert cfg["seed"] * TRAIN_SEED_STRIDE + cfg["n_episodes"] <= VAL_SEED_OFFSET

    def episode(ep, rng):
        return prepare(make_prices(cfg["mercado"], cfg["n_steps"], cfg["seed"] * TRAIN_SEED_STRIDE + ep))
    return (_synthetic(cfg["mercado"], "calibracion", cfg["n_steps"], 0),
            _synthetic(cfg["mercado"], "test", cfg["n_steps"], cfg["n_test"]), episode)


# ---------------------------------------------------------------------------
# Entrenamiento y test con caché
# ---------------------------------------------------------------------------
def evaluate(agent, encoder, series, cfg):
    old_rng = getattr(agent, "rng", None)
    if old_rng is not None:
        agent.rng = np.random.default_rng(200_000 + cfg.get("seed", 0))
    try:
        rows = [episode_metrics(*run_episode_anexo(p, agent, encoder, cfg["coste"], cfg["recompensa"])[1:])
                for p in series]
    finally:
        if old_rng is not None:
            agent.rng = old_rng
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def fingerprint():
    h = hashlib.sha256(extras_fingerprint().encode())
    h.update((ROOT / "src" / "extras_anexo.py").read_bytes())
    return h.hexdigest()


def cache_id(cfg):
    """Huella de configuración, código y datos. `experimento` no cuenta: se comparten entrenamientos."""
    payload = {**{k: v for k, v in cfg.items() if k != "experimento"}, "code": fingerprint()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def train_eval(cfg, recompute=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    cid = cache_id(cfg)
    js, npz = CACHE / f"{cid}.json", CACHE / f"{cid}.npz"
    if not recompute and js.exists() and npz.exists():
        return json.loads(js.read_text(encoding="utf-8"))
    calib, test, episode = sources(cfg)
    encoder = fit_subset_encoder(calib, cfg["estado"])
    agent = make_agent(cfg, encoder)
    visits = np.zeros(encoder.n_states, dtype=np.int64)
    rewards = np.empty(cfg["n_episodes"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.perf_counter()
    for ep in range(cfg["n_episodes"]):
        rewards[ep], *_ = run_episode_anexo(episode(ep, rng), agent, encoder, cfg["coste"], cfg["recompensa"],
                                            learn=True, greedy=False, visits=visits)
        agent.end_episode()
    elapsed = time.perf_counter() - t0
    before = agent.Q.copy()
    df = evaluate(agent, encoder, test, cfg)
    assert np.array_equal(agent.Q, before), "El test no puede modificar Q"
    row = {**{k: v for k, v in cfg.items() if k != "experimento"},
           **summarize(f"{cfg['algoritmo']}_{cfg['estado']}", cfg["mercado"], df, rewards, elapsed),
           "id": cid, "n_estados": encoder.n_states, "cobertura": float(np.mean(visits > 0)),
           "visitas_mediana": float(np.median(visits))}
    np.savez_compressed(npz, Q=agent.Q, rewards=rewards, visits=visits,
                        **{f"edges{i}": e for i, e in enumerate(encoder.edges)})
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
    js.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return clean


def _job(args):
    return train_eval(*args)


def run_anexo(configs, name="03_anexo_semillas", n_jobs=6, recompute=False):
    """Entrena y evalúa cada configuración distinta una sola vez; cada experimento recibe sus filas."""
    unique = {cache_id(c): c for c in configs}
    start, results = time.perf_counter(), {}
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i, row in enumerate(pool.map(_job, [(c, recompute) for c in unique.values()]), 1):
            results[row["id"]] = row
            if i % 25 == 0 or i == len(unique):
                print(f"Anexo: {i}/{len(unique)} entrenamientos · {time.perf_counter()-start:.0f} s", flush=True)
    df = pd.DataFrame([{"experimento": c["experimento"], **results[cache_id(c)]} for c in configs])
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def baseline(cfg, name):
    """Buy & hold o no operar sobre las mismas series de test y con la misma comisión."""
    calib, test, _ = sources(cfg)
    agent = BuyHoldAgent() if name == "buy_hold" else CashAgent()
    return summarize(name, cfg["mercado"], evaluate(agent, fit_subset_encoder(calib, "base"), test, cfg))


def load_q(cfg):
    """Tabla Q, visitas y bordes del entrenamiento `cfg` (lo entrena si no está en caché)."""
    row = train_eval(cfg)
    with np.load(CACHE / f"{row['id']}.npz") as arr:
        return {k: arr[k].copy() for k in arr.files}


def exposure_map(cfg):
    """Exposición deseada por celda (tendencia × MA30 × volatilidad) de la política greedy.

    0 = fuera, 1 = dentro, 0,5 = mantiene si ya está dentro pero no entra (histéresis por la comisión),
    NaN = celda sin visitas en ninguna de las dos posiciones.
    """
    arr = load_q(cfg)
    shape = tuple(len(arr[f"edges{i}"]) + 1 for i in range(len(COLUMNS[cfg["estado"]])))
    greedy = arr["Q"].argmax(axis=1).reshape(*shape, 2)
    flat_in = greedy[..., 0] == 2      # sin posición: compra
    held_in = greedy[..., 1] != 0      # con posición: no vende
    out = np.where(flat_in & held_in, 1.0, np.where(held_in, 0.5, 0.0))
    seen = arr["visits"].reshape(*shape, 2).sum(axis=-1) > 0
    return np.where(seen, out, np.nan)
