# rl-trading-agents · Agentes de RL tabular para trading

Práctica en grupo del Máster MIAX (módulo de Aprendizaje por Refuerzo): Q-Learning y SARSA tabulares
para invertir en un único activo sobre mercados sintéticos (tendencia + ciclo, browniano geométrico,
reversión a la media y regímenes alcista/bajista), comparados con un agente aleatorio y con buy & hold.

## Enunciado (resumen)

- **Entorno:** `SimpleTradingEnv` del taller. El estado es `(tendencia, posición)`, las acciones son
  vender / mantener / comprar, la comisión es del 0,1 % y la recompensa es la variación del precio
  con posición abierta menos los costes.
- **Mínimos:**
  1. Q-Learning y SARSA en los 4 mercados, con una serie nueva en cada episodio y evaluación
     promediada sobre series de test.
  2. Comparación con el aleatorio y con buy & hold, razonando si la estrategia es la esperada.
  3. Ajuste de α, ε (con decaimiento), γ y número de episodios.
  4. Tabla comparativa de rendimiento y tiempo de convergencia.
- **Entrega:** PDF de **2 páginas como máximo** (pruebas, resultados y conclusiones), una por grupo.
  - Antes del **martes 27/10:** un representante manda un email al profesor (luis.lago@uam.es) para
    fijar el orden de las presentaciones.
  - Antes de la clase del **jueves 29/10:** primera versión por email.
  - **29/10, en clase:** presentación y discusión de 15 minutos.
  - **29/10, 23:59:** entrega final.
- **Evaluación:** los mínimos dan el aprobado (50 %). El otro 50 % se reparte entre ir más allá de los
  mínimos (10 %), la calidad de presentación (20 %), la justificación del diseño (10 %) y el análisis
  crítico (10 %).

## Estructura

```
├── README.md              # este fichero (hace también de índice)
├── requirements.txt
├── docs/
│   ├── DECISIONES.md      # registro de decisiones D-X.n
│   ├── teoria/            # síntesis del material de clase (extraído de raw/)
│   └── raw/               # material bruto de clase (NO versionado)
├── src/                   # código común: única fuente de verdad
│   ├── env.py             # SimpleTradingEnv (copia del taller) + discretize()
│   ├── markets.py         # 4 generadores del taller (+ extras) y make_prices(market, n, seed)
│   ├── agents.py          # RandomAgent, BuyHoldAgent, QLearningAgent, SarsaAgent
│   └── experiment.py      # train(), evaluate(), summarize(), save_results(), gráficas
├── notebooks/
│   ├── 00_entorno_y_baselines.ipynb
│   ├── 01_qlearning.ipynb
│   ├── 02_sarsa.ipynb
│   ├── 03_extras.ipynb
│   └── 04_comparativa.ipynb   # solo lee results/: tabla final y figuras
└── results/
    ├── NN_*.csv           # resultados de cada notebook (esquema común)
    ├── figures/           # figuras del informe
    └── informe.md         # entregable → PDF de 2 páginas
```

## Cómo trabajar

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

- **Empezad por `notebooks/00_entorno_y_baselines.ipynb`**: explica el problema con figuras, mide los baselines y
  cierra las decisiones comunes. Se lee en GitHub ya ejecutado.
- Todos los notebooks empiezan con la misma celda de setup, que importa de `src/` (las figuras usan el estilo común
  de `src/style.py`).
- **Contrato común:**
  - `src/env.py` y `src/markets.py` son copia literal del taller y no se modifican; los mercados
    extra se añaden al final de `markets.py`.
  - Todos los agentes implementan `act(state, greedy)`, `update(s, a, r, s2, a2, done)` y
    `end_episode()`. Los tabulares reciben sus bins al crearse
    (`QLearningAgent(edges=quantile_edges(mercado, N_BINS))`, D-1.1), y `run_episode` respeta el orden de
    actualización de cada algoritmo (D-2.1).
  - El protocolo (series, semillas de train y test) está en `src/experiment.py` (D-0.1).
  - Cualquier cambio en `src/` se avisa al grupo y se registra en `docs/DECISIONES.md`.
- **Líneas de trabajo en paralelo:** 01 (Q-Learning), 02 (SARSA) y 03 (extras). Cada una guarda
  `results/NN_<nombre>.csv` con `save_results()`, que usa las columnas `agente, mercado,
  retorno_mediana, retorno_medio, std, sharpe, max_drawdown, n_trades, episodios_convergencia, tiempo_s`.
- **04:** concatena los CSV, construye la tabla del informe y guarda las figuras en
  `results/figures/`.

## Documentación

- [`docs/DECISIONES.md`](docs/DECISIONES.md): decisiones del proyecto.
- `docs/teoria/` (síntesis del material de clase, con la fuente exacta citada en cada concepto):
  - `notebook_partida.md`: enunciado y rúbrica, el entorno con sus sutilezas, los 4 generadores y
    qué estrategia cabe esperar en cada uno, los comentarios del profesor en clase (con minuto),
    ideas para ir más allá y una checklist ligada a la rúbrica. **Empezad por aquí.**
  - `00_fundamentos_mdp.md`: RL, bandits, MDP, retorno y γ, funciones de valor, ecuaciones de
    Bellman y GridWorld.
  - `01_dp_y_montecarlo.md`: iteración de políticas y de valores, Monte Carlo (ES, on y
    off-policy), ε-greedy y qué enseña cada notebook de clase.
  - `02_td_sarsa_qlearning.md`: TD(0), SARSA, Q-Learning, cliff walking, hiperparámetros y las
    reglas de `update()` que hay que implementar. **Es el núcleo de la tarea.**
- `docs/raw/` (local, no versionado): enunciado `taller_trader.ipynb`; slides `slides1.pdf`
  (fundamentos y MDP), `slides2.pdf` (programación dinámica y métodos tabulares) y `slides3.pdf`
  (TD learning); notebooks de clase (`gridworld`, `frozen_lake`, `taxi`, `gambler`, `blackjack`,
  `clase_20261023`); y la transcripción de la clase del 23/10 (`26 10 23.pdf`).

## Resultados hasta ahora

Mediana del retorno en 200 series de test (`results/00_baselines.csv`, 00 §6). El Q-Learning es el de calibración
del 00 (α = 0,001, γ = 0,9, 500 episodios, 5 bins por cuantiles), no el agente final del 01.

| Mercado | Aleatorio | Buy & hold | Q-Learning (calibración) | Política aprendida |
|---|---|---|---|---|
| 1 · Tendencia + ciclo | x4,3 | x33,6 | **x343** | Sigue la tendencia |
| 2 · Browniano | −19,4 % | −3,5 % | 0 % | No opera (no hay señal) |
| 3 · Reversión a la media | −17,2 % | −0,3 % | **+11,8 %** | Compra tras caídas y vende en máximos |
| 4 · Regímenes | −22,0 % | −12,5 % | 0 % | No opera (no encuentra el seguimiento de tendencia) |

Claves (00 §8): **α tiene que ser pequeño** (≈ 10⁻³; con 0,1 el agente no opera en 2–4, D-2.2), en 2–4 "no operar" ya
iguala o bate a buy & hold, y el reto es el mercado 4.

## Estado

- [x] Estructura del repositorio y código común
- [x] Documentación de teoría (`docs/teoria/`)
- [x] 00 Entorno y baselines (D-0.1, D-1.1 y D-0.3 cerradas)
- [ ] 01 Q-Learning
- [ ] 02 SARSA
- [ ] 03 Extras
- [ ] 04 Comparativa
- [ ] Informe PDF
