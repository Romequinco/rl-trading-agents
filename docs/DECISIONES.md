# Registro de decisiones del proyecto

Formato ligero tipo ADR. ID `D-X.n`: `X` es el bloque (0 general, 1 entorno/estado,
2 algoritmos, 3 entregable) y `n` un correlativo. El estado es `ABIERTA` o `CERRADA`.

---

### D-0.1 · Protocolo común de entrenamiento y evaluación

- **Fecha:** 2026-10-26
- **Decisión provisional:** series de `N_STEPS = 500` pasos. En entrenamiento, cada episodio usa una
  serie nueva del mismo mercado con semilla `0, 1, 2, ...`. Las de test son `N_TEST = 50` series
  con semillas desde `100_000`, que no se ven nunca al entrenar, y se evalúan con política greedy y
  sin aprender. Está implementado en `src/experiment.py` (`train`, `evaluate`) y todos los
  notebooks lo usan.
- **Justificación:** el taller pide series distintas por episodio y resultados promediados sobre
  varias series de test. Fijar las semillas hace comparables los números de todos los miembros.
- **Pendiente:** confirmar `N_TEST` en el 00 (que la desviación de la media sea pequeña).
- **Estado:** ABIERTA

### D-0.2 · Código común en `src/` y un notebook por línea de trabajo

- **Fecha:** 2026-10-26
- **Decisión:** el entorno y los generadores son copia literal del taller (`src/env.py`,
  `src/markets.py`). Los agentes comparten la interfaz `act / update / end_episode`
  (`src/agents.py`) y el protocolo vive en `src/experiment.py`. Cada notebook guarda
  `results/NN_*.csv` con el esquema `RESULT_COLUMNS`, y el 04 solo concatena.
- **Justificación:** la entrega es un PDF, no un notebook autocontenido. Tres personas en paralelo
  necesitan una sola fuente de verdad y notebooks separados (los `.ipynb` no se mezclan bien en git).
  Un cambio en `src/` se avisa al grupo y se anota aquí.
- **Estado:** CERRADA

### D-1.1 · Discretización del estado

- **Fecha:** 2026-10-26
- **Contexto:** la observación es `(tendencia, posición)` con la tendencia continua en [-1, 1]. Los
  métodos tabulares necesitan estados discretos.
- **Decisión provisional:** 5 bins de tendencia con bordes `[-0.02, -0.005, 0.005, 0.02]`, por 2
  posiciones, lo que da 10 estados (`src/env.py`, `TREND_EDGES`).
- **Evidencia inicial (semilla 0, percentiles 5/50/95 de la tendencia):** en tendencia+ciclo,
  -0,23 / 0,01 / 0,26; en browniano, -0,04 / 0,00 / 0,04; en reversión a la media,
  -0,04 / 0,00 / 0,04; en regímenes, -0,05 / 0,00 / 0,03. Con bordes fijos de ±0,02 el mercado 1
  queda casi siempre en los bins extremos. El motivo es que `generar_precios` tiene un ruido del 10 %
  por paso, frente al 1–2,5 % del resto.
- **Alternativas:** (a) bordes fijos comunes; (b) bordes por cuantiles de cada mercado, calculados
  solo con series de train (`quantile_edges()` en `src/experiment.py`); (c) más bins.
- **Implementación:** cada agente tabular lleva sus bordes
  (`QLearningAgent(edges=quantile_edges("regimenes"))`), y `run_episode` discretiza con los bordes
  del agente. Así la tabla Q, el entrenamiento y el test siempre usan los mismos bins.
- **Evidencia de (b)** (20 series de train, 5 bins): bordes ≈ `[-0.11, -0.02, 0.06, 0.15]` en
  tendencia+ciclo y ≈ ±0,005 / ±0,02 en los otros tres. Coinciden con los fijos salvo en el mercado
  1. En una prueba rápida (300 episodios, código temporal) Q-Learning y SARSA entrenan sin errores
  con estos bordes.
- **Propuesta:** (b) con 5 bins como opción por defecto, y probar (c) en el 01/02 como
  hiperparámetro.
- **Estado:** ABIERTA (se cierra en el 00)

### D-0.3 · Métricas y convergencia

- **Fecha:** 2026-10-26
- **Decisión:** en test se mide el retorno final medio y su desviación, el Sharpe por paso (sin
  anualizar), el max drawdown y el número de operaciones efectivas. La convergencia es el primer
  episodio en que la media móvil (ventana 50) de la recompensa de entrenamiento alcanza el 90 % de
  la mejora entre su valor inicial y el final (`convergence_episode()`). Sobre una curva sintética
  con τ = 150, da 334 episodios (el valor teórico es ≈345).
- **Problema:** en tendencia+ciclo el retorno compuesto con ruido del 10 % da medias enormes y muy
  dispersas (buy & hold +200x de media, std 477x). En ese mercado conviene añadir la mediana o el
  log-retorno para comparar.
- **Estado:** ABIERTA

### D-2.1 · Orden de actualización y estado terminal

- **Fecha:** 2026-10-26
- **Decisión:** se sigue el pseudocódigo de slides3. En SARSA (p31) se elige A' antes de
  actualizar; en Q-Learning (p36) se actualiza y después se elige la acción con la Q ya
  actualizada. Esto importa porque, con solo 10 estados, `s2 == s` es frecuente. En el último paso
  (`terminated`), el objetivo es solo `r`, porque Q(terminal, ·) = 0.
- **Nota:** el final del episodio es en realidad un límite de tiempo y no un estado terminal del
  MDP. Como afecta a un único paso de unos 500, se acepta.
- **Estado:** CERRADA

### D-3.1 · Formato del informe

- **Fecha:** 2026-10-26
- **Decisión:** el informe se redacta en `results/informe.md`, junto a las tablas y figuras, y se
  exporta a PDF de 2 páginas como máximo.
- **Pendiente:** herramienta de exportación y plantilla.
- **Estado:** ABIERTA
