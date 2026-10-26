# Registro de decisiones del proyecto

Formato ligero tipo ADR. ID `D-X.n`: `X` es el bloque (0 general, 1 entorno/estado,
2 algoritmos, 3 entregable) y `n` un correlativo. El estado es `ABIERTA` o `CERRADA`.

---

### D-0.1 · Protocolo común de entrenamiento y evaluación

- **Fecha:** 2026-10-26
- **Decisión:** series de `N_STEPS = 500` pasos. En entrenamiento, cada episodio usa una serie nueva
  del mismo mercado con semilla `0, 1, 2, ...`. Las de test son **`N_TEST = 200`** series con
  semillas desde `100_000`, que no se ven nunca al entrenar, y se evalúan con política greedy y sin
  aprender. Está implementado en `src/experiment.py` (`train`, `evaluate`) y todos los notebooks lo
  usan.
- **Justificación:** el taller pide series distintas por episodio y resultados promediados sobre
  varias series de test. Fijar las semillas hace comparables los números de todos los miembros.
- **Evidencia (00 §5):** con 50 series, la mediana de buy & hold se mueve hasta 10 puntos
  (browniano: +6,9 % con 50 y −3,5 % con 200). Con 200, el error estándar baja a ≈ ±4 puntos
  (±0,6 en reversión a la media). Evaluar 200 series cuesta solo unos segundos por agente.
- **Estado:** CERRADA (2026-10-26, en el 00)

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
- **Evidencia de la calibración (00 §7.2):** Q-Learning con α = 0,001, γ = 0,9 y 500 episodios. La
  mediana en test de las opciones fijos-5 / cuantiles-5 / cuantiles-10 es:
  - mercado 1: x386 / x343 / x404 (buy & hold, x33);
  - reversión a la media: +14,8 % / +11,8 % / +13,6 % (buy & hold, −0,3 %);
  - browniano y regímenes: 0 % con las tres opciones (no opera).

  Las diferencias están dentro del ruido de un solo entrenamiento.
- **Decisión:** (b), **bins por cuantiles de cada mercado con `N_BINS = 5`** (10 estados),
  `quantile_edges(m, N_BINS)`. Rinde igual que las otras opciones, tiene la mitad de estados que 10
  bins y se adapta a la escala de cualquier mercado. Los bordes fijos solo funcionan en 2–4 por
  coincidencia y en el mercado 1 saturan: 89 % de los pasos en los bins extremos (00 §4). 10 bins,
  la sugerencia del profesor, queda como hiperparámetro para el 01/02.
- **Estado:** CERRADA (2026-10-26, en el 00)

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
- **Decisión (2026-10-26):** la métrica principal es la **mediana del retorno**
  (`retorno_mediana`, añadida al esquema común), con la media ± std al lado. Con 200 series de
  test, en el mercado 1 la media de buy & hold es +19 925 % y la mediana +3 262 %.
- **Estado:** CERRADA

### D-2.2 · La tasa de aprendizaje α debe ser pequeña

- **Fecha:** 2026-10-26
- **Evidencia (00 §7.1, Q-Learning con cuantiles-5, γ = 0,9, 500 episodios):** con α = 0,1 o 0,01
  el agente **no opera nunca** en los mercados 2–4. Con α = 0,001 aprende en reversión a la media
  (+11,8 %, 61 operaciones, política de comprar tras caídas y vender en máximos). La explicación es
  que, con α constante, Q es una media exponencial cuyo ruido es ≈ σ√(α/2): 0,45 % con σ = 2 % y
  α = 0,1, frente a una ventaja real de ≈ 0,1 % por paso.
- **Implicación para 01/02:** barrer α en torno a 10⁻³ (o con decaimiento) antes que el resto de
  hiperparámetros. Además, en el mercado de regímenes ningún α encuentra el seguimiento de
  tendencia esperado: el agente se queda siempre fuera (0 %, frente a −12,5 % de buy & hold). Ese es
  el reto principal.
- **Estado:** ABIERTA (guía para el ajuste del 01/02)

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
