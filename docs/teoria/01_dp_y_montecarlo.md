# Programación dinámica, métodos tabulares y Monte Carlo

> **Fuentes:** slides2.pdf (L. F. Lago Fernández, "Programación Dinámica y Métodos Tabulares", MIAX-14, p1–p54) · `docs/raw/frozen_lake.ipynb`, `taxi.ipynb`, `gambler.ipynb`, `blackjack.ipynb`.
>
> **Convenciones:** "(slides2 pN)" = página N · "(frozen_lake.ipynb, celda ...)" = celda del notebook · "Nota" = contexto propio que no aparece en las fuentes.

---

## 1. Métodos tabulares (slides2 p4–p9)
Se busca la solución de las ecuaciones de optimalidad de Bellman de $V_*$ y $Q_*$ (slides2 p4; ver `00_fundamentos_mdp.md`). Los métodos **tabulares** son aplicables cuando estados y acciones son lo bastante pocos como para guardar las funciones de valor en tablas o matrices. Hay tres familias (slides2 p5):

| Propiedad | Programación dinámica (DP) | Monte Carlo (MC) | Diferencia temporal (TD) |
|---|---|---|---|
| Requiere modelo $p(s',r\mid s,a)$ | Sí (model-based) | No (model-free) | No (model-free) |
| Bootstrapping | Sí | No | Sí |
| Tipo de tarea | Episódica y continua | Solo episódica | Episódica y continua |

(slides2 p9; descripción por familia en p6–p8). *Bootstrapping* = actualizar la estimación de un estado usando estimaciones de otros estados. TD es híbrido: aprende de la experiencia como MC y hace bootstrapping como DP (slides2 p8).

---

## 2. Programación dinámica
Calcula funciones de valor óptimas **dado un modelo perfecto** del fMDP: la función $p(s',r\mid s,a)$ se supone conocida. Una vez hallado $V_*$ o $Q_*$, obtener la política óptima es trivial (slides2 p10).

### 2.1 Evaluación de políticas (slides2 p11–p14)
Calcular $V^\pi$ para una $\pi$ arbitraria resolviendo la ecuación de expectativa de Bellman (sistema lineal; existe solución única si $\gamma<1$ o todo episodio termina en tiempo finito) (slides2 p11).

**Iterative policy evaluation** (slides2 p12): sucesión $V_0,V_1,\dots$ con $V_0$ arbitraria y
$$V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V_k(s')\,]$$
Si $V_0=0$ en terminales, $V_k\to V^\pi$ cuando $k\to\infty$.
- Dos implementaciones: dos arrays ($V_k$ y $V_{k+1}$) o un único array **in-place**, donde parte de las actualizaciones usa valores ya modificados en el mismo barrido (slides2 p13).
- **Criterio de parada:** $\max_s|V_{k+1}(s)-V_k(s)|<\theta$ con $\theta>0$ (slides2 p13).

Pseudocódigo (slides2 p14): inicializar $V$ (0 en terminales); repetir {$\Delta\leftarrow0$; para cada $s$: $v\leftarrow V(s)$; $V(s)\leftarrow\sum_a\pi(a\mid s)\sum_{s',r}p\,[r+\gamma V(s')]$; $\Delta\leftarrow\max(\Delta,|v-V(s)|)$} hasta $\Delta<\theta$.

### 2.2 Mejora de políticas (slides2 p15–p18)
Dada $V^\pi$ y $\pi$ determinista, se prueba en $s$ una acción $a\ne\pi(s)$ y luego se sigue $\pi$: su valor es $Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma V^\pi(s')]$ (slides2 p15).
- **Teorema de mejora de políticas:** si $Q^\pi(s,\pi'(s))\ge V^\pi(s)\ \forall s$, entonces $V^{\pi'}(s)\ge V^\pi(s)\ \forall s$ (slides2 p16).
- **Política greedy:** $\pi'(s)=\arg\max_aQ^\pi(s,a)=\arg\max_a\sum_{s',r}p\,[r+\gamma V^\pi(s')]$ cumple el teorema por construcción (slides2 p17).
- Si $V^{\pi'}=V^\pi$, se verifica la ecuación de optimalidad de Bellman y ambas son óptimas (slides2 p18).

### 2.3 Iteración de políticas (slides2 p19–p22)
Alterna evaluación y mejora a partir de una $\pi_0$ arbitraria:
$$\pi_0\xrightarrow{\text{Eval}}V^{\pi_0}\xrightarrow{\text{Mejora}}\pi_1\xrightarrow{\text{Eval}}V^{\pi_1}\to\dots\to\pi_*\xrightarrow{\text{Eval}}V_*$$
Garantiza la política óptima en un número finito de iteraciones (slides2 p19).

Pseudocódigo (slides2 p20–p22): inicializar $V$ y $\pi$ arbitrarias; bucle {evaluación completa (repetir barridos con $V(s)\leftarrow\sum_{s',r}p(s',r\mid s,\pi(s))[r+\gamma V(s')]$ hasta $\Delta<\theta$); mejora (para cada $s$: $a_{old}\leftarrow\pi(s)$, $\pi(s)\leftarrow\arg\max_a\sum p[r+\gamma V(s')]$, si $a_{old}\ne\pi(s)$ entonces `policy_stable=false`); si `policy_stable`, parar}.
- **Criterio de parada:** política estable (ninguna acción cambia en un barrido de mejora).

### 2.4 Iteración de valores (slides2 p23–p25)
Inconveniente de policy iteration: cada evaluación requiere varios barridos hasta converger, muy costoso (slides2 p23). *Value iteration* hace **un único barrido** por iteración, combinando evaluación y mejora en la regla de optimalidad:
$$V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V_k(s')\,]\quad\forall s$$
Converge a $V_*$ bajo las mismas condiciones que garantizan su existencia (slides2 p24).

Pseudocódigo (slides2 p25): inicializar $V$ (0 en terminales); repetir {$\Delta\leftarrow0$; para cada $s$: $v\leftarrow V(s)$, $V(s)\leftarrow\max_a\sum p[r+\gamma V(s')]$, $\Delta\leftarrow\max(\Delta,|v-V(s)|)$} hasta $\Delta<\theta$; después extraer $\pi(s)=\arg\max_a\sum p[r+\gamma V(s')]$.

### 2.5 Limitaciones de DP (slides2 p29)
1. Exige que el entorno sea un fMDP con **dinámica completamente conocida**; en la práctica el entorno no suele ser markoviano o $p(s',r\mid s,a)$ no se conoce.
2. Cada iteración recorre **todos** los estados; con muchos estados es inviable almacenar y barrer la tabla.

---

## 3. Monte Carlo (MC)
Estiman funciones de valor y políticas óptimas a partir de **experiencia muestreada** (secuencias de estados, acciones y recompensas), sin conocer la dinámica. Solo valen para tareas episódicas; la estimación y la política se cambian **al final de cada episodio** (primero evaluación, luego mejora) (slides2 p30).

### 3.1 Evaluación MC (slides2 p31–p33)
Con episodios $S_0,A_0,R_1,\dots,S_{T-1},A_{T-1},R_T$ generados con $\pi$, el retorno es $G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}$. Como $V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$, el **promedio empírico** de los retornos tras visitar $s$ converge a $V^\pi(s)$ (slides2 p31).
- **First-visit:** promedia solo el retorno tras la primera visita a $s$ en cada episodio. **Every-visit:** tras cada visita. Ambas son consistentes (slides2 p32).
- Pseudocódigo first-visit (slides2 p33): generar episodio con $\pi$; $G\leftarrow0$; recorrer $t=T-1,\dots,0$ hacia atrás con $G\leftarrow\gamma G+R_{t+1}$; si $S_t$ no aparece antes en el episodio, añadir $G$ a `Returns(S_t)` y $V(S_t)\leftarrow$ promedio.

### 3.2 MC frente a DP (slides2 p34–p35)
- MC: $V(S_t)\leftarrow\text{promedio}(G_t)$, estimaciones independientes por estado, **sin bootstrapping**. DP: $V(S_t)\leftarrow\mathbb{E}_\pi[R_{t+1}+\gamma V(S_{t+1})]$ con la dinámica conocida (slides2 p34).
- **Sesgo:** MC no tiene sesgo; DP está sesgado al inicio por la elección de $V_0$.
- **Varianza:** MC tiene alta varianza porque $G_t$ acumula una secuencia larga de acciones y transiciones estocásticas.
- **Markov:** MC no se ve afectado si los estados no cumplen estrictamente la propiedad de Markov.
- **Coste:** en MC, el coste de estimar un estado no depende del número de estados (slides2 p35).

### 3.3 Estimar $Q$ y necesidad de exploración (slides2 p36–p38)
Sin modelo, $V(s)$ no basta para elegir acciones, así que el objetivo primario es $Q(s,a)$: en cada episodio se mide el retorno tras la (primera) visita de cada par $(s,a)$ (slides2 p36). Con una $\pi$ determinista solo se observa una acción por estado y el resto de pares queda sin explorar (slides2 p37). Dos soluciones (slides2 p38):
1. **Exploring starts:** cada episodio comienza en un par $(S_0,A_0)$ elegido al azar entre todos los de $\mathcal S\times\mathcal A$.
2. **Políticas soft:** $\pi(a\mid s)>0\ \forall s,a$.

### 3.4 Control MC (slides2 p39–p41)
Igual que en DP, alterna evaluación de $Q^{\pi_k}$ y mejora greedy: $\pi_{k+1}(s)=\arg\max_aQ^{\pi_k}(s,a)$. Con exploring starts y evaluación completa, el teorema de mejora de políticas aplica y $\pi_{k+1}\ge\pi_k$; si son iguales, son óptimas (slides2 p39–p40). En la práctica se relaja: $Q$ no es exacta y se alterna evaluación y mejora **episodio a episodio**, mejorando la política solo en los estados visitados (slides2 p41).

### 3.5 MC con exploring starts (MC ES) (slides2 p42, p47)
Pseudocódigo (slides2 p42): inicializar $Q$, $\pi$ y `Returns(s,a)`; bucle {elegir $(S_0,A_0)$ al azar con probabilidad $>0$ para todos los pares; generar episodio con $\pi$; recorrer hacia atrás con $G\leftarrow\gamma G+R_{t+1}$; en la primera visita de $(S_t,A_t)$: añadir $G$, $Q\leftarrow$ promedio, $\pi(S_t)\leftarrow\arg\max_aQ(S_t,a)$}.
- Los retornos de un par se promedian sin importar con qué política se observaron. Converge a la política óptima, pero exploring starts es poco realista porque no permite aprender de interacción real (slides2 p47). Para evitarlo hacen falta políticas soft.

### 3.6 MC on-policy (slides2 p48–p49)
Evalúa y mejora **la misma política** que usa para interactuar. Para que no se vuelva determinista, la mejora se restringe a políticas $\epsilon$-greedy respecto a $Q$:
$$\pi(a\mid s)=\begin{cases}1-\epsilon+\frac{\epsilon}{|A(s)|}&a=\arg\max_bQ(s,b)\\ \frac{\epsilon}{|A(s)|}&\text{resto}\end{cases}$$
El algoritmo es el de MC ES con la política siempre $\epsilon$-greedy y sin exploring starts; converge a la mejor política **entre las $\epsilon$-greedy** (slides2 p48). Pseudocódigo en slides2 p49: como MC ES, pero tras actualizar $Q(S_t,A_t)$ se reasigna $\pi(a\mid S_t)$ con el reparto anterior y $A^*=\arg\max_aQ(S_t,a)$ (empates arbitrarios).

### 3.7 MC off-policy (slides2 p50–p52)
Usa dos políticas: la **objetivo** $\pi$ (la que se optimiza, acaba determinista y óptima) y la de **comportamiento** $b$ (estocástica y exploratoria, genera los datos). Requiere **cobertura**: $\pi(a\mid s)>0\Rightarrow b(a\mid s)>0$ (slides2 p50).
- **Muestreo por importancia:** el retorno se pondera por $\rho_{t:T-1}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}$ (las probabilidades de transición se cancelan), de modo que el retorno ponderado es $\rho_{t:T-1}G_t$ (slides2 p51).
- Pseudocódigo (slides2 p52): inicializar $Q$ arbitraria, $C(s,a)=0$, $\pi(s)=\arg\max_aQ(s,a)$; por episodio con $b$: $G\leftarrow0$, $W\leftarrow1$; hacia atrás: $G\leftarrow\gamma G+R_{t+1}$, $C\leftarrow C+W$, $Q\leftarrow Q+\frac{W}{C}[G-Q]$, $\pi(S_t)\leftarrow\arg\max_aQ$; si $A_t\ne\pi(S_t)$, **romper**; $W\leftarrow W/b(A_t\mid S_t)$.

### 3.8 Limitaciones de MC (slides2 p53)
1. **No aprende online:** hay que esperar al final del episodio para actualizar; inaplicable con episodios muy largos o continuos.
2. **Varianza** de $G_t$ crece con la longitud del episodio, exigiendo muchos episodios.
3. **Off-policy ineficiente con trayectorias largas:** si $\pi$ difiere de $b$ en algún paso, $\rho$ se anula y se descarta el resto del episodio.

Ejercicio propuesto (slides2 p54): programar MC on-policy y off-policy (el texto dice "On-Policy MC y On-Policy MC", probable errata) y probarlos en FrozenLake-v1, Taxi-v4 y Blackjack-v1, comparando resultados.

---

## 4. Exploración $\epsilon$-greedy: resumen
- Regla $\epsilon$-greedy: probabilidad $1-\epsilon$ greedy, $\epsilon$ al azar entre todas las acciones (slides1 p17; en MC, slides2 p48). La mejor acción recibe en total $1-\epsilon+\epsilon/|A|$.
- Alternativas para garantizar exploración sostenida: exploring starts o políticas soft (slides2 p38); o separar comportamiento y objetivo (off-policy, slides2 p50).
- Dilema de fondo: ver `00_fundamentos_mdp.md` §2 (slides1 p13).

---

## 5. Qué ilustra cada notebook

### 5.1 `frozen_lake.ipynb`: value iteration en un MDP estocástico
- Entorno `FrozenLake-v1` 4×4 (16 estados, 4 acciones: 0 izquierda, 1 abajo, 2 derecha, 3 arriba), con `is_slippery=False` en el notebook. Recompensa +1 al llegar a la meta, 0 en el resto; agujeros y meta son terminales; episodio de máximo 100 iteraciones (celda "Dinámica del entorno").
- El modelo se lee de `env.unwrapped.P`, con `P[s][a] = [(prob, siguiente_estado, recompensa, terminal), ...]` (celda "Probabilidades de transición").
- `value_iteration(env, gamma=0.99, theta=1e-8)`: $V$ inicial `np.zeros(n_states)`; barrido con $q(a)=\sum \text{prob}\,(r+\gamma V(s'))$ **sin bootstrapping si `done`** (valor 0 del terminal); $V[s]=\max_aq$; se actualiza **in-place** y se para cuando `delta < theta`. La política se extrae con `argmax` de los $q$ (celda "Value iteration").
- Resultado: tablero con $V_*$ y flechas de $\pi_*$ (`plot_value_and_policy`). En slides1 p44–p45 se compara `is_slippery=False` y `True`.

### 5.2 `taxi.ipynb`: el mismo algoritmo con 500 estados
- `Taxi-v4`: 500 estados (25 posiciones del taxi × 5 del pasajero, incluido en el taxi × 4 destinos) y 6 acciones (sur, norte, este, oeste, recoger, dejar) (celda "Dinámica del entorno").
- La función `value_iteration` es idéntica a la de FrozenLake (`gamma=0.99`, `theta=1e-8`, inicio en ceros, parada por `delta`). Se dibujan $V_*$ y $\pi_*$ en la rejilla 5×5 para un pasajero y destino fijos; el índice se calcula como `((fila*5+col)*5 + pasajero)*4 + destino`. Incluye grabación de vídeo de 10 episodios siguiendo `policy` (`RecordVideo`).
- Ilustra que DP escala a cientos de estados cuando `P` está disponible en el entorno.

### 5.3 `gambler.ipynb`: value iteration con política no trivial
- Problema de Sutton & Barto 4.3: fMDP episódico no descontado; estado = capital $s\in\{1,\dots,99\}$; acción = apuesta $a\in\{1,\dots,\min(s,100-s)\}$; recompensa 1 solo al llegar a 100. Probabilidad de cara **`p = 0.4`** (celdas de descripción y de `p`).
- Inicialización: `v = np.zeros(100)` (el índice 0 solo sirve para que cada índice coincida con su capital). Se implementan dos versiones de la actualización: copia (`update_value`) e **in-place** (`update_value_in_place`, la usada). Se ejecutan **32 iteraciones fijas** (no hay umbral $\theta$) y se grafican las iteraciones 1, 2, 3 y 32 (slides2 p27).
- La política se obtiene con `get_policy` y una **tolerancia `1e-9`** en la comparación (`pv > maxv + 1e-9`); sin ella aparecen picos espurios por acciones empatadas en valor. El autor lo comprueba recorriendo las acciones en orden inverso, lo que cambia la política elegida entre empates (celdas finales). Lección práctica: al hacer `argmax` de valores casi iguales, el desempate afecta a la política resultante.

### 5.4 `blackjack.ipynb`: Monte Carlo sin modelo
- `gym.make("Blackjack-v1", sab=True)` (reglas del libro). Estado = (suma del jugador, carta visible del dealer, as usable); acciones 0 = plantarse y 1 = pedir carta; recompensa +1/−1/0 al final (slides2 p44).
- **Evaluación MC** de la política "plantarse con 20 o 21, pedir en otro caso": **500 000 episodios**, sin descuento (se usa la recompensa final como retorno de todos los estados visitados), valor por **media incremental** `V ← V*(n-1)/n + G/n` con contador `num_visits`; es una variante every-visit en la práctica, ya que se recorre `state_list` completo (celda "Evaluación Monte Carlo"). Resultado similar al libro (superficies 3D con y sin as usable, slides2 p45).
- **Monte Carlo ES:** **5 000 000 episodios**; política inicial aleatoria (0 o 1 con probabilidad 0,5 por estado, para sumas 4–21); la primera acción de cada episodio es aleatoria (exploring starts) y el resto sigue la política; $Q(s,a)$ por media incremental con `num_visits`; tras cada episodio, la política se actualiza **solo en los estados visitados**: `0` si $Q(s,0)>Q(s,1)$, `1` en caso contrario (empates → pedir carta). Se dibuja la política como mapa de calor (celda "Monte Carlo ES"). La figura de slides2 p46 corresponde a 500 000 iteraciones.
- Ilustra el control MC por episodios con exploring starts, visto en §3.5.

---

## Aplicación al taller
Marco del taller: estado discreto (bin de tendencia, posición) con 20 estados (10 bins × 2 posiciones, D-1.1), acciones {vender, mantener, comprar}, recompensa = variación de precio con posición menos comisión del 0,1 %, episodios de ~500 pasos con serie nueva por episodio.

- **Por qué no DP.** Policy iteration y value iteration (slides2 p10, p20–p25) necesitan $p(s',r\mid s,a)$, que `env.unwrapped.P` entrega en FrozenLake/Taxi y en el Gambler's problem se conoce por construcción. En el taller la recompensa depende de una serie de precios aleatoria, sin modelo analítico; es la limitación 1 de slides2 p29. Con solo 20 estados el coste de barrer la tabla no sería el problema, sino la ausencia de modelo.
- **Por qué métodos model-free.** MC y TD aprenden de episodios muestreados y no requieren el modelo (slides2 p7–p9). Nota: se pueden generar tantos episodios como se quiera (serie nueva por episodio), que es el escenario para el que están pensados.
- **Por qué TD (Q-Learning y SARSA) frente a MC.** MC espera al final del episodio y tiene alta varianza con episodios largos (slides2 p53, limitaciones 1 y 2); con ~500 pasos esto penaliza. TD actualiza en cada paso con bootstrapping (slides2 p8). Q-Learning es el análogo off-policy y SARSA el on-policy del par visto en MC (slides2 p48–p52).
- **Exploración.** Con una política greedy sobre $Q$ algunos pares (posición, tendencia, acción) nunca se visitarían (slides2 p37). Se usa $\epsilon$-greedy (política soft, slides2 p38, p48). Nota: exploring starts no es aplicable, porque no se puede forzar al mercado a empezar en un estado y acción arbitrarios.
- **Factor $\gamma$.** Las recompensas tras cada paso son pequeñas y las decisiones tienen efecto sostenido por la posición abierta; $\gamma$ regula cuánto pesa ese efecto futuro (slides1 p26). Con el episodio finito de ~500 pasos $\gamma=1$ es matemáticamente admisible (slides2 p11), pero queda como hiperparámetro a justificar.
- **Detalles de implementación reutilizables.** Inicializar $Q$ en ceros (como `np.zeros` en los notebooks); media incremental como en blackjack.ipynb; tolerancia en comparaciones de empates como en gambler.ipynb (`1e-9`) al extraer la política; y no hacer bootstrapping tras el estado terminal, como en `if done` de frozen_lake.ipynb (el valor del terminal es 0).
