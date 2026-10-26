# Empezad por aquí: enunciado, entorno y lo que dijo el profesor

> Fuentes: `docs/raw/taller_trader.ipynb` (enunciado y código del taller; solo código y markdown,
> no las salidas) y la transcripción automática de la clase del 23/10 (`docs/raw/26 10 23.pdf`).
> Las marcas *(transcripción ~hh:mm)* son el minuto aproximado de la grabación. La transcripción es
> muy ruidosa: las citas son paráfrasis salvo las frases cortas entre comillas. Lo marcado como
> **Nota/propuesta** es del redactor, no del enunciado ni del profesor.

## 1. Qué pide exactamente la entrega

| Requisito | Fuente | Dónde se hace |
|---|---|---|
| Implementar un agente de RL que aprenda a invertir en un entorno simulado con **un único activo** | enunciado §Objetivo | `src/agents.py` |
| **Ej. 1:** al menos **Q-Learning y SARSA**, con al menos los **4 generadores**. Muchos episodios, **cada uno con una serie distinta** del mismo generador. Test con **series nuevas** del mismo generador, **promediando** sobre varias | enunciado §Ejercicios | 01, 02 · D-0.1 |
| **Ej. 2:** analizar la solución de cada mercado, compararla con el **aleatorio** y con **buy & hold**, y **razonar si la estrategia es la esperada** | enunciado §Ejercicios | 00 (baselines), 01–02, informe |
| **Ej. 3:** ajustar **α**, **ε** (probablemente con decaimiento), **γ** y el **número de episodios** | enunciado §Ejercicios | 01, 02 |
| **Ej. 4:** **tabla** algoritmos + baselines × mercado, con al menos **rendimiento** y **tiempo de convergencia** | enunciado §Ejercicios | 04 · D-0.3 |
| **PDF de 2 páginas como máximo**: pruebas realizadas, resultados y conclusiones. **Una entrega por grupo** (de TFM) | enunciado §Entrega | `results/informe.md` · D-3.1 |
| Un **representante** manda un email al profesor **antes del martes 27/10** (para fijar el orden de presentación, que se comunica ese día) | enunciado §Entrega | **urgente** (hoy es 26/10) |
| **Primera versión** por email a **luis.lago@uam.es** antes de la clase del **jueves 29/10** | enunciado §Entrega | informe v1 |
| **Presentación y discusión** de **15 min como máximo** por grupo en esa clase. Los comentarios sirven para mejorar la versión final | enunciado §Entrega | — |
| **Entrega final: 29/10/2026, 23:59** | enunciado §Entrega | informe final |

**Evaluación** (enunciado §Evaluación):

| Bloque | Peso |
|---|---|
| Ejercicios y pruebas propuestos (mínimos): garantizan el aprobado | 50 % |
| Pruebas que vayan **más allá** de los mínimos | 10 % |
| **Calidad de presentación** de los resultados | 20 % |
| **Justificación** de las decisiones de diseño | 10 % |
| **Análisis crítico** de las soluciones | 10 % |

**Consecuencias prácticas:**

- El 50 % no mínimo pesa sobre todo en presentación (20 %). En dos páginas, cada figura y la tabla
  tienen que leerse solas.
- La "justificación" y el "análisis crítico" (20 % entre ambos) se apoyan en `DECISIONES.md`:
  cada decisión que salga en el informe debe tener su D-x con evidencia.
- El enunciado no fija métricas más allá de "rendimiento" y "tiempo de convergencia". Las demás
  (Sharpe, drawdown, operaciones) son decisión nuestra (D-0.3).

## 2. El entorno `SimpleTradingEnv` (tal como está en el código)

### 2.1 Observación, acciones y dinámica

| Elemento | Código del taller | Comentario |
|---|---|---|
| Observación | `[clip(trend, -1, 1), position]`, `float32`, `Box([-1,0],[1,1])` | Tendencia **continua** + posición 0/1 |
| Tendencia | `window = prices[t-5 : t+1]` (**6 precios**, incluido el actual); `trend = (p_t − mean(window)) / mean(window)` | El comentario del código dice "últimos 5 pasos", pero la ventana tiene 6 precios. Es la desviación relativa respecto a una media móvil corta, **no una pendiente** |
| Acciones | `Discrete(3)`: 0 = vender, 1 = mantener, 2 = comprar | Vender sin activo o comprar con él equivale a mantener |
| Inicio | `reset()`: `current_step = 5`, `position = 0`, `cash = balance = 1000`, `shares = 0` | Empieza siempre fuera del mercado |
| Ejecución | Compra/venta **al precio `prev_price = p_t`**, todo el capital (all-in / all-out) | `p_t` es el último precio que ya entra en la observación: no hay mirada al futuro |
| Comisión | `cost_rate = 0.001` (0,1 %) sobre el efectivo al comprar y sobre el valor de venta al vender | — |
| Recompensa | `−0.001` si opera + `(p_{t+1} − p_t)/p_t` si **termina el paso con el activo** | El paso de compra ya cobra la variación; el de venta, no |
| Balance | `cash + shares · p_{t+1}` con activo; `cash` sin él. Va en `info["balance"]` | Valor de mercado: la posición abierta al final **no paga la comisión de venta** |
| Fin | `terminated = current_step >= len(prices) − 1`; `truncated = False` siempre | Con `n = 500` hay **494 decisiones** por episodio |

### 2.2 Sutilezas que afectan al diseño

1. **La tendencia es continua → hay que discretizar** para los métodos tabulares. El profesor lo
   dijo explícitamente (§3). Su rango depende mucho del mercado: en el mercado 1 los percentiles
   5/95 son ≈ ±0,23 y en los otros ≈ ±0,04 (D-1.1). Por eso unos bordes fijos comunes no sirven.
2. **Recompensa y balance no miden lo mismo.** La comisión se descuenta igual en ambos en términos
   relativos (−0,1 %). Pero la recompensa **suma retornos simples** sin ponderar por el capital,
   mientras que el balance **compone** (multiplica). Maximizar Σr solo equivale a maximizar el
   balance final para retornos pequeños. Con el ruido del 10 % por paso del mercado 1, la diferencia
   es grande (efecto de la volatilidad sobre el retorno compuesto). Además, con γ < 1 se maximiza
   una suma descontada, no el retorno del episodio.
3. **El mercado 1 es distinto de los otros tres:** ruido N(0, 0,1) por paso (10 %), frente a un
   1–2,5 % del resto. Sus medias de retorno final salen enormes y muy dispersas (D-0.3): conviene
   la mediana o el log-retorno.
4. **`reset()` no genera precios nuevos.** La serie se pasa al constructor (y `max_steps` se fija
   ahí), así que "una serie distinta por episodio" exige crear el entorno con una serie nueva en
   cada episodio. Los generadores usan `np.random` global sin semilla: el repo usa
   `make_prices(market, n, seed)` para reproducibilidad (D-0.1).
5. **Estado muy pequeño:** `(bin, posición)` da 2 × nº de bins estados. Con pocos estados,
   `s' == s` es frecuente, y eso hace que importe el orden de actualización (D-2.1).
6. **El final es un límite de tiempo, no un estado terminal del MDP** (D-2.1).
7. **Acciones redundantes:** en cada estado una de las tres acciones equivale a "mantener". La
   tabla Q aprende valores para ellas igualmente.
8. **El estado no es (necesariamente) markoviano:** la tendencia resume 6 precios. El profesor
   insistió en que el mundo financiero probablemente no lo es y en que depende de cómo se defina el
   estado (§3). Es un buen argumento para el análisis crítico.

### 2.3 Los 4 generadores (con sus parámetros por defecto)

| # | Función | Parámetros | Qué cabe esperar del agente *(Nota/propuesta, a comprobar)* |
|---|---|---|---|
| 1 | `generar_precios` (tendencia + ciclo) | `t = linspace(0, 50, n)`; retorno = `0.0005·t + 0.05·sin(t) + N(0, 0.1)`; precio = `100·cumprod(1+ret)` | Hay deriva creciente y un ciclo en los **retornos** (periodo 2π en `t`, ≈ 63 pasos con `n = 500`), así que es aprendible. El profesor dice que logró superar al aleatorio y a buy & hold (§3). **Ojo:** la deriva y el periodo dependen de `n_steps` |
| 2 | `generar_precios_browniano` (GBM) | `start_price=100, mu=1e-4, sigma=0.02`; log-retornos N(μ − σ²/2, σ) | Paseo aleatorio, sin información en la tendencia. Lo esperable es **no batir a buy & hold**: operar solo paga comisiones |
| 3 | `generar_precios_reversion_media` | `mean_price=100, kappa=0.05, volatility=0.02` (ruido aditivo N(0, 2)); suelo 0,1 | Oscila en torno a 100 (desviación estacionaria ≈ 6). Buy & hold ≈ 0 de media. Una tendencia negativa puede anticipar rebote, pero el estado **no ve la distancia a 100** |
| 4 | `generar_precios_regimenes` | Cambio de régimen con p = 0,02 por paso (≈ 50 pasos por régimen); alcista N(0,001, 0,01), bajista N(−0,0015, 0,025); empieza alcista | Estar dentro en los regímenes alcistas y fuera en los bajistas, con el retraso que introduce la media de 6 precios |

El enunciado pide las pruebas "al menos" con estas cuatro y permite añadir otras funciones o una
serie real.

### 2.4 Baselines del taller

- **Aleatorio:** `env.action_space.sample()` en cada paso.
- **Buy & hold:** compra en `prices[5]` con comisión (`1000 · 0.999 / prices[5]` acciones) y
  valora a mercado hasta el final, sin vender. Es coherente con el entorno: tampoco paga comisión
  de salida.
- La figura del taller tiene dos paneles: precio con puntos verdes (con activo) y rojos (sin él), y
  balance del agente frente a buy & hold y al capital inicial. Es un buen formato para el informe.

## 3. Comentarios y recomendaciones del profesor (clase 23/10)

La clase del 23/10 cubrió fundamentos, bandits, MDP, programación dinámica y los ejemplos de
gridworld, FrozenLake y Taxi. El taller se presentó al final (~04:22–04:41). En esta transcripción
**no se explican TD, SARSA ni Q-Learning**: para eso, `slides3`.

### 3.1 Sobre el taller y la entrega

| Tema | Qué dijo (paráfrasis) | Minuto |
|---|---|---|
| Estado del enunciado | Los ejercicios "no están del todo" definidos y los refinaría. **Manda el enunciado final del notebook** | ~04:22, ~04:37 |
| Entorno | Problema "muy sencillo" con apariencia financiera: decidir cuándo entrar y cuándo salir. Para crear otro entorno basta con extender `gym.Env` (`reset`, `_get_obs`, `step`) | ~04:23 |
| Discretización | Con un método tabular hay que discretizar la tendencia, por ejemplo "partís en 10 cachitos", "o los que sean" | ~04:27 |
| Comisión | Se mete para hacerlo un poco más realista | ~04:27 |
| Precios reales | Si queréis, os descargáis una serie real y se la pasáis al constructor. Él no los usará | ~04:31 |
| Mercado 1 | Es posible aprender: él entrenó un agente mejor que el aleatorio y que buy & hold | ~04:35 |
| Objetivo | El aleatorio gana algo porque la serie es claramente alcista. El objetivo es **mejorar a buy & hold** | ~04:36 |
| Lectura de la política | Lo ideal es tener el activo cuando sube y no tenerlo cuando baja (puntos verdes/rojos) | ~04:36 |
| Entrenamiento | No entrenar siempre con la misma serie: una serie distinta por episodio de la misma función o, como **reto mayor**, de **funciones distintas**, para que el agente funcione bien en todas | ~04:37 |
| Análisis | Para cada generador, decir qué estrategia esperaríais y ver si el agente la encuentra. Si encuentra otra, entenderla. Puso el ejemplo de FrozenLake, donde la política óptima se dirige al agujero y él no lo había previsto | ~04:38 (FrozenLake ~04:07) |
| Uso de IA | Le preocupa menos que programéis los algoritmos vosotros: le parece bien usar una IA generativa | ~04:38 |
| Lo que valora | Que **entendáis la solución** y seáis capaces de **analizar la convergencia** ("explicadme qué algoritmo converge más rápido"). Preguntará sobre esto en la discusión | ~04:39 |
| Formato v1 | Un resumen de "una cara o 2 caras" con los resultados, para discutir. Mencionó Word, pero **el enunciado pide PDF** | ~04:40 |
| Sesión del jueves | Se dedica a revisar la solución con cada grupo. Por grupos, para poder hablar con todos en 2 h y media | ~04:40 |

### 3.2 Conceptos de clase que afectan a las decisiones

- **ε con decaimiento:** tras el ejemplo de bandits dijo que tendría mucho sentido empezar con un
  ε alto e ir bajándolo cuando ya se ha probado lo suficiente *(transcripción ~01:10)*. Respalda
  el ej. 3.
- **No llevar ε a 0 si el entorno cambia:** con un ejemplo financiero (activos cuyos retornos
  cambian en el tiempo) dijo que, si el mercado cambia, hay que seguir explorando
  *(transcripción ~01:21)*. Es relevante para el mercado de regímenes y para justificar un ε mínimo.
  En test evaluamos en greedy (D-0.1), lo que conviene justificar.
- **Markov:** sobre si el mundo financiero es markoviano dijo "me temo que no", y que depende de
  cómo definas el estado *(transcripción ~01:34)*. Encaja con el análisis crítico del estado
  `(tendencia, posición)`.
- **γ:** se descuenta porque las recompensas lejanas son inciertas y para que la suma converja en
  tareas continuas *(transcripción ~01:40)*. Aquí el episodio es finito (494 pasos), así que γ
  cercano a 1 es viable. Lo que hace γ es fijar cuánto pesa el futuro frente a la comisión
  inmediata.
- **Recompensas intermedias:** en ajedrez pueden despistar del objetivo a largo plazo
  *(transcripción ~00:14)*. Conviene tenerlo en cuenta si se modifica la recompensa.

### 3.3 Temas sobre los que la transcripción no dice nada

Valores concretos de α, γ, ε o del número de episodios; cómo medir la convergencia; qué métricas
usar (Sharpe, drawdown, etc.); cuántas series de test; cómo repartir el informe ni qué valora en
él más allá de entender la solución y la convergencia; errores típicos; SARSA frente a Q-Learning.
Todo eso son decisiones del grupo y deben justificarse en `DECISIONES.md`.

## 4. Ideas para "ir más allá" (10 %)

**Del enunciado o del profesor:**

- **Más generadores** o **una serie de precios real** (enunciado §Generadores; profesor ~04:31).
- **Entrenar con series de generadores distintos** y evaluar en todos: un único agente robusto
  (profesor ~04:37, "un reto mayor").
- **Otros algoritmos vistos en clase**, además de Q-Learning y SARSA (enunciado: "al menos").
- **Granularidad de la discretización** (profesor ~04:27: 10 bins "o los que sean"), alternativa
  (c) de D-1.1.
- **Comparar la velocidad de convergencia** de los algoritmos y explicarla (profesor ~04:39).

**Nota/propuesta (del redactor):**

- **Generalización fuera de distribución:** entrenar en un mercado y evaluar en otro (matriz 4×4).
- **Varias semillas de entrenamiento** por configuración, para separar la variabilidad del
  entrenamiento de la del test.
- **Visualizar la política aprendida** (mapa de Q o de la acción greedy por bin y posición) y
  contrastarla con la estrategia esperada de §2.3.
- **Ampliar el estado**: tendencias a varias ventanas, o distancia a una media larga para la
  reversión. Así se ve el límite de la observación del taller sin tocar `src/env.py` (habría que
  hacerlo en un entorno derivado, en `03`).
- **Sensibilidad a la comisión** (0 %, 0,1 %, 0,5 %): cómo cambia el número de operaciones.
- **Recompensa alternativa** (log-retorno del balance), por la sutileza 2 de §2.2, comparada con la
  original.

## 5. Checklist final (ligada a la rúbrica)

| Rúbrica | Comprobación | Dónde |
|---|---|---|
| Logística | Email del representante enviado antes del **27/10** | — |
| Logística | v1 enviada a luis.lago@uam.es antes de la clase del **29/10** | `results/informe.md` → PDF |
| Logística | PDF final de **2 páginas como máximo**, una entrega por grupo, **29/10 23:59** | D-3.1 |
| Mínimos (50 %) | Q-Learning y SARSA entrenados en los 4 mercados, con una serie nueva por episodio | 01, 02 · D-0.1 |
| Mínimos | Test con series nuevas, resultados promediados (media ± std) | `evaluate()` · D-0.1 |
| Mínimos | Comparación con aleatorio y buy & hold en cada mercado | 00, 04 |
| Mínimos | Para cada mercado: estrategia esperada frente a la encontrada, razonada | 01, 02, informe |
| Mínimos | Ajuste de α, ε (decaimiento), γ y número de episodios documentado | 01, 02 |
| Mínimos | Tabla algoritmos + baselines × mercado con rendimiento y convergencia | 04 · D-0.3 |
| Más allá (10 %) | Al menos una o dos pruebas de §4, con resultado y conclusión | 03 |
| Presentación (20 %) | Tabla legible, figuras de precio/posición y balance, curvas de aprendizaje; todo cabe en 2 páginas | 04, `results/figures/` |
| Presentación | Guion de 15 min: entorno → diseño → resultados → análisis | — |
| Justificación (10 %) | Discretización (D-1.1), protocolo (D-0.1), métricas y convergencia (D-0.3) y orden de actualización (D-2.1) cerrados y citados | `DECISIONES.md` |
| Análisis crítico (10 %) | Límites del estado (no markoviano, sin distancia a la media), recompensa frente a balance (§2.2), dispersión del mercado 1 y qué algoritmo converge antes y por qué | informe |
| Discusión con el profesor | Cada miembro sabe explicar Q-Learning frente a SARSA (on/off-policy) y la convergencia observada | slides3 |
