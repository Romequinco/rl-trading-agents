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

En Windows (PowerShell), desde la carpeta del proyecto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name rl-trading-agents --display-name "Python (RL Trading)"
.\.venv\Scripts\python.exe -m jupyterlab notebooks/
```

Si el entorno ya está creado, solo hace falta ejecutar el último comando. No es necesario
activarlo ni cambiar la política de ejecución de PowerShell. En el cuaderno, selecciona el
kernel **Python (RL Trading)**; en VS Code también puedes seleccionar `.venv\Scripts\python.exe`.

En otros sistemas:

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

- Los barridos de hiperparámetros usan `run_grid()` (en paralelo y con caché: reabrir un notebook no
  reentrena). Para forzarlo, `RECALCULAR = True` en la celda de setup.
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

Mediana del retorno en las **200 series de test** (`results/00_baselines.csv`, `results/01_qlearning.csv`,
`results/02_sarsa.csv`). Para los agentes es la mediana de 5 semillas, con hiperparámetros ajustados en validación
(D-0.4, D-2.3).

| Mercado | Aleatorio | Buy & hold | **Q-Learning** | **SARSA** | Estrategia aprendida (las dos coinciden) |
|---|---|---|---|---|---|
| 1 · Tendencia + ciclo | x4,3 | x33,6 | **x502** | **x471** | Sigue la tendencia (entra en bins altos, sale en 1–3) |
| 2 · Browniano | −19,4 % | −3,5 % | **0 %** | **0 %** | No opera (correcto: no hay señal) |
| 3 · Reversión a la media | −17,2 % | −0,3 % | **+27,0 %** | **+27,0 %** | Compra tras la caída más fuerte y vende tras la subida más fuerte |
| 4 · Regímenes | −22,0 % | −12,5 % | **0 %** | **0 %** | No opera: es lo óptimo con este estado (D-2.4) |

Claves:
- α = 0,001 en todos los mercados; con 0,1 el agente no aprende (D-2.2).
- γ depende del mercado: 0 en el 1 y 0,5 en el 3 (01 §4).
- Q-Learning converge en ≈ 100–260 episodios; SARSA, algo más despacio (≈ 200–360). Con la misma configuración, 189
  frente a 203 (mercado 1) y 260 frente a 282 (reversión) (D-2.5).
- SARSA es más conservador mientras explora, pero la diferencia es pequeña porque un error cuesta solo un 0,1 % y ε
  decae a 0,01; con ε mínimo 0,05 sí opera bastante menos (02 §5).
- En el mercado 4, ninguna de las 70 reglas simples sobre el estado (tendencia o reversión) bate a no operar, así que el límite está en el estado
  y no en el algoritmo (01 §9).

## Comparación directa: base frente a estado enriquecido

El estado enriquecido añade **distancia a MA30 y volatilidad20** al estado base de tendencia corta y posición.
La comparación usa las dos variantes del experimento 03: mismos hiperparámetros, 800 episodios por entrenamiento,
cinco semillas y las mismas historias de test. No se compara contra las configuraciones ajustadas de 01–02.
Los valores son los resúmenes de retorno de `results/03_extras.csv`; **Δ = enriquecido − base**, en puntos porcentuales.

| Mercado | Algoritmo | Base | Enriquecido | Δ retorno (pp) |
|---|---|---:|---:|---:|
| Reversión a la media | Q-Learning | +18.1% | +42.8% | +24.6 |
| Reversión a la media | SARSA | +18.1% | +43.8% | +25.6 |
| Regímenes | Q-Learning | +0.0% | +7.8% | +7.8 |
| Regímenes | SARSA | +0.0% | +7.5% | +7.5 |
| Tendencia persistente | Q-Learning | +104.5% | +123.8% | +19.3 |
| Tendencia persistente | SARSA | +103.0% | +128.7% | +25.7 |
| Difusión + saltos | Q-Learning | +0.0% | +0.0% | +0.0 |
| Difusión + saltos | SARSA | +0.0% | +0.0% | +0.0 |
| S&P 500 · real | Q-Learning | +45.9% | +28.3% | -17.5 |
| S&P 500 · real | SARSA | +45.9% | +29.9% | -16.0 |

Mejora el retorno en reversión, regímenes y tendencia persistente; en difusión con saltos no hay cambio.
En el S&P 500, el enriquecido reduce el retorno y también la caída máxima: Q-Learning pasa de 18,9 % a 11,4 %
de drawdown y SARSA de 18,9 % a 12,8 %. Añadir información no mejora todas las medidas en todos los mercados.
La serie real es un único recorrido de test de 2024–2025; las semillas no son historias independientes.

La [comparación completa](results/03_base_vs_enriquecido.md) incluye las tablas de rentabilidad y drawdown y un gráfico
de las diferencias. También se muestra en el apartado **«Base frente a estado enriquecido»** de
`notebooks/03_extras.ipynb`, que regenera el [CSV de diferencias](results/03_base_vs_enriquecido.csv), el informe y la figura.
Estas diferencias entre resúmenes no son la mediana de diferencias emparejadas de `03_estado_deltas.csv`.

## Estado

- [x] Estructura del repositorio y código común
- [x] Documentación de teoría (`docs/teoria/`)
- [x] 00 Entorno y baselines (D-0.1, D-1.1 y D-0.3 cerradas)
- [x] 01 Q-Learning (D-0.4, D-2.2, D-2.3 y D-2.4)
- [x] 02 SARSA (D-2.5)
- [x] 03 Extras (estado enriquecido, mercados extra y serie real)
- [ ] 04 Comparativa
- [ ] Informe PDF

## Extras del notebook 03

`notebooks/03_extras.ipynb` compara el estado base (20 celdas) y uno enriquecido con
distancia a MA30 y volatilidad20 (180), tanto con Q-Learning como con SARSA. Incluye
los mercados de reversión y regímenes, dos generadores extra (tendencia persistente y
difusión con saltos) y una serie real del S&P 500 de FRED, con cortes cronológicos
2017–2021 / 2022–2023 / 2024–2025. Los datos se guardan localmente con su procedencia
y huella en `data/`; ejecutar el cuaderno no requiere descargar precios.

La ejecución completa usa cinco semillas, 800 episodios y 200 historias sintéticas de
test; la real es una sola historia de test por política. Guarda `results/03_extras.csv`
con el esquema común, además de resultados por semilla/episodio, diferencias emparejadas,
cobertura, bloques reales, conclusiones y figuras. Las tablas Q se cachean con una huella
de la configuración, código y datos. `NB_QUICK=1` ejecuta la comprobación corta con
salidas `_quick`; no sustituye los resultados completos.

El código del 03 vive en `src/extras.py`. La contabilidad del entorno y los algoritmos
originales se conservan. Los extras solo amplían el registro de mercados cuando se
importa ese módulo, para no alterar los barridos de 00–02. Decisiones: D-1.2, D-1.3 y D-0.5.

Pruebas desde la raíz: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
