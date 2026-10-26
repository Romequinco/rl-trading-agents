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

- Todos los notebooks empiezan con la misma celda de setup, que importa de `src/`.
- **Contrato común:**
  - `src/env.py` y `src/markets.py` son copia literal del taller y no se modifican; los mercados
    extra se añaden al final de `markets.py`.
  - Todos los agentes implementan `act(state, greedy)`, `update(s, a, r, s2, a2, done)` y
    `end_episode()`. Los tabulares reciben sus bins al crearse
    (`QLearningAgent(edges=quantile_edges(mercado))`, D-1.1), y `run_episode` respeta el orden de
    actualización de cada algoritmo (D-2.1).
  - El protocolo (series, semillas de train y test) está en `src/experiment.py` (D-0.1).
  - Cualquier cambio en `src/` se avisa al grupo y se registra en `docs/DECISIONES.md`.
- **Líneas de trabajo en paralelo:** 01 (Q-Learning), 02 (SARSA) y 03 (extras). Cada una guarda
  `results/NN_<nombre>.csv` con `save_results()`, que usa las columnas `agente, mercado,
  retorno_medio, std, sharpe, max_drawdown, n_trades, episodios_convergencia, tiempo_s`.
- **04:** concatena los CSV, construye la tabla del informe y guarda las figuras en
  `results/figures/`.

## Documentación

- [`docs/DECISIONES.md`](docs/DECISIONES.md): decisiones del proyecto.
- `docs/teoria/`: pendiente (síntesis de las slides, notebooks y transcripción de clase).
- `docs/raw/` (local, no versionado): enunciado `taller_trader.ipynb`; slides `slides1.pdf`
  (fundamentos y MDP), `slides2.pdf` (programación dinámica y métodos tabulares) y `slides3.pdf`
  (TD learning); notebooks de clase (`gridworld`, `frozen_lake`, `taxi`, `gambler`, `blackjack`,
  `clase_20261023`); y la transcripción de la clase del 23/10 (`26 10 23.pdf`).

## Resultados hasta ahora

Pendiente.

## Estado

- [x] Estructura del repositorio y código común
- [ ] Documentación de teoría (`docs/teoria/`)
- [ ] 00 Entorno y baselines (cerrar D-0.1 y D-1.1)
- [ ] 01 Q-Learning
- [ ] 02 SARSA
- [ ] 03 Extras
- [ ] 04 Comparativa
- [ ] Informe PDF
