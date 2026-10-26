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
  queda casi siempre en los bins extremos.
- **Alternativas:** bordes fijos comunes, bordes por cuantiles de cada mercado (calculados en
  train) o más bins.
- **Estado:** ABIERTA (se cierra en el 00)

### D-3.1 · Formato del informe

- **Fecha:** 2026-10-26
- **Decisión:** el informe se redacta en `results/informe.md`, junto a las tablas y figuras, y se
  exporta a PDF de 2 páginas como máximo.
- **Pendiente:** herramienta de exportación y plantilla.
- **Estado:** ABIERTA
