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
- **Decisión:** (b)+(c), **bins por cuantiles de cada mercado con `N_BINS = 10`** (20 estados),
  `quantile_edges(m, N_BINS)`.
  - **Por qué 10:** es lo que recomendó el profesor en clase ("partís en 10 cachitos", transcripción
    ~04:27). Además, en la calibración rinde igual o mejor que 5: es la mejor opción en el mercado 1
    (x404) y la segunda en reversión (+13,6 %).
  - **Por qué cuantiles:** cada bin recibe ≈ 10 % de los pasos y la discretización se adapta sola
    a la escala de cualquier mercado (o de una serie real). Los bordes fijos solo funcionan en 2–4
    por coincidencia y en el mercado 1 saturan: 89 % de los pasos en los dos bins extremos (00 §4).
  - **Coste asumido:** 20 estados en lugar de 10, así que cada estado se visita la mitad y el
    aprendizaje puede ser algo más lento. Con 500 episodios × 494 pasos no ha sido un problema.
- **Historial:** el 2026-10-26 se fijó primero `N_BINS = 5`, porque rendía igual con menos estados.
  El grupo prefirió seguir la recomendación del profesor, ya que la evidencia no distingue entre
  las dos opciones.
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

### D-0.4 · Validación separada del test para ajustar hiperparámetros

- **Fecha:** 2026-10-26
- **Decisión:** los hiperparámetros se eligen con `N_VAL = 100` series de **validación** (semillas desde `50_000`;
  `evaluate(..., split="val")`). Las 200 de test (D-0.1) solo se usan una vez, para el resultado final. Con la
  semilla de entrenamiento `s`, las series de entrenamiento son `s * 10_000 + episodio`, de modo que nunca se
  cruzan con las de validación (`run_config` lo comprueba).
- **Justificación:** elegir mirando el test daría un resultado optimista. En el 01 la validación sale algo por
  encima del test (reversión: +34 % frente a +27 %), lo esperable al quedarse con el máximo de muchas
  configuraciones.
- **Implementación:** `run_grid(configs, nombre)` en `src/experiment.py` entrena en paralelo y guarda una caché en
  `results/<nombre>.csv` (versionado) y `results/cache/` (curvas y tablas Q, no versionado).
- **Estado:** CERRADA

### D-2.2 · La tasa de aprendizaje α debe ser pequeña

- **Fecha:** 2026-10-26
- **Evidencia del 00 (§7.1):** con α = 0,1 el agente no opera en los mercados 2–4. El ruido de Q es
  ≈ σ√(α/2) ≈ 0,45 %, frente a una ventaja de ≈ 0,1 % por paso.
- **Evidencia del 01 (§3, 3 semillas, validación):** α = 0,001 es el elegido en los cuatro mercados. Entre 0,0005 y
  0,01 los resultados forman una meseta casi plana; con 0,03 empeora y con 0,1 se hunde (mercado 1: x14 frente a
  x527; reversión: no opera).
- **Decisión:** α = 0,001 para Q-Learning. Para SARSA (02) se barre el mismo rango.
- **Evidencia del 02 (§3):** SARSA prefiere α = 0,0005 en los mercados 1 y 3 (reversión: +34 % frente a +22,6 % en
  validación). Con 0,03 deja de operar en reversión y con 0,1 se hunde en el mercado 1 (x19). Coherente con un objetivo
  más ruidoso (usa la acción muestreada, no el máximo).
- **Estado:** CERRADA para Q-Learning (01) y SARSA (02)

### D-2.3 · Ajuste de hiperparámetros por mercado, por etapas y con parsimonia

- **Fecha:** 2026-10-26
- **Decisión:**
  - Se ajusta **por mercado y por etapas**: α → γ → ε (decaimiento / mínimo) → episodios.
  - Cada configuración se entrena con **3 semillas** y se puntúa con la mediana entre semillas de la mediana en
    validación.
  - **Regla de parsimonia:** un valor nuevo solo sustituye al de partida si le gana en la mayoría de las semillas
    (comparación semilla a semilla) y tiene mejor mediana.
  - El resultado final se mide con **5 semillas** en test.
- **Justificación:** los mercados son muy distintos, y la regla de parsimonia evita elegir por ruido. En la
  prueba rápida, sin la regla, el browniano habría elegido α = 0,01 por un +3,4 % que salía de 1 sola
  operación.
- **Resultado del 01 (Q-Learning):**
  - Hiperparámetros: mercado 1 → α 0,001, γ 0, ε 0,98 / 0,05, 500 episodios; mercado 3 → α 0,001, γ 0,5,
    ε 0,99 / 0,01, 500 episodios; mercados 2 y 4 → valores de partida (no opera con ninguna combinación).
  - Test: x502 · 0 % · +27,0 % · 0 % (buy & hold: x33,6 · −3,5 % · −0,3 % · −12,5 %).
- **Estado:** CERRADA para el 01. El 02 debe usar el mismo procedimiento.

### D-2.5 · SARSA: resultado y comparación con Q-Learning

- **Fecha:** 2026-10-26
- **Decisión:** SARSA se ajusta con el procedimiento de D-2.3 (mismo punto de partida, mismas rejillas, validación, 3
  semillas, parsimonia). La pregunta "¿qué algoritmo converge antes?" se responde con un **duelo a configuración idéntica**
  (α 0,001, γ 0,9, ε 0,99/0,01, 1 000 episodios, mismas 3 semillas, `results/02_duelo.csv`), no con la columna
  `episodios_convergencia` del test final, que mezcla algoritmo e hiperparámetros.
- **Hiperparámetros:** mercado 1 → α 0,0005, γ 0, ε 0,99 / 0,01, 1 000 episodios; mercado 3 → α 0,0005, γ 0,9,
  ε 0,99 / 0,01, 500 episodios; mercados 2 y 4 → valores de partida (no opera con ninguna combinación).
- **Test (02 §7):** x471 · 0 % · +27,0 % · 0 %. Varias semillas reproducen exactamente la política de Q-Learning
  (x501,3 y +27,0 %): con 20 estados × 3 acciones los dos llegan a la misma tabla de decisiones.
- **Convergencia (02 §6 y §9):** Q-Learning converge antes (duelo: 189 frente a 203 episodios en el mercado 1, 260
  frente a 282 en reversión). SARSA tiene algo más de recompensa online y opera menos (56 frente a 66 operaciones en el
  mercado 1), como predice *cliff walking*, pero la diferencia es pequeña porque un error de exploración solo cuesta un
  0,1 % y ε decae a 0,01. Con ε mínimo 0,05, SARSA opera un tercio menos en reversión (22 frente a 32) y una semilla cae
  a +2,8 %; Q-Learning apenas cambia.
- **Estado:** CERRADA

### D-2.4 · Mercado 4: no operar es lo óptimo con este estado

- **Fecha:** 2026-10-26
- **Evidencia (01 §9):** se probaron 70 reglas fijas sobre el estado. Las 35 de tendencia ("entrar si bin ≥ e,
  salir si bin ≤ x") tienen su mejor resultado en −8,6 %; las 35 de reversión ("entrar si bin ≤ e, salir si bin ≥ x"),
  en −9,2 %. Las mejores baten a buy & hold en las mismas series de validación (−19,4 %), pero ninguna se acerca al
  0 % de no operar. Es una prueba fuerte, aunque no una demostración, de que no hay ventaja simple con este estado.
- **Conclusión:** Q-Learning no falla. La tendencia de 6 precios no identifica el régimen (llega tarde y el
  régimen bajista tiene una volatilidad 2,5 veces mayor).
- **Implicación:** en el 03 se probará un estado ampliado (tendencia más larga o volatilidad).
- **Estado:** CERRADA (análisis). El estado ampliado queda ABIERTO para el 03.

### D-2.1 · Orden de actualización y estado terminal

- **Fecha:** 2026-10-26
- **Decisión:** se sigue el pseudocódigo de slides3. En SARSA (p31) se elige A' antes de
  actualizar; en Q-Learning (p36) se actualiza y después se elige la acción con la Q ya
  actualizada. Esto importa porque, con solo 20 estados, `s2 == s` es frecuente. En el último paso
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
