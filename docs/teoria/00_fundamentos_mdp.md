# Fundamentos del aprendizaje por refuerzo y MDP

> **Fuentes:** slides1.pdf (L. F. Lago Fernández, "Fundamentos de RL y Procesos de Decisión de Markov", MIAX-14, p1–p48) · `docs/raw/gridworld.ipynb` · `docs/raw/clase_20261023.ipynb`.
>
> **Convenciones:** "(slides1 pN)" = página N de las diapositivas · "(gridworld.ipynb, ...)" = celda o sección del notebook · "Nota" = contexto propio que no aparece en las fuentes.

---

## 1. Qué es el aprendizaje por refuerzo
**Idea.** El RL es un paradigma de aprendizaje automático en el que un agente aprende, interactuando con su entorno, a asociar situaciones (estados) con acciones para maximizar una señal de recompensa numérica a largo plazo (slides1 p4).
- Nadie le dice qué acción tomar: la descubre por prueba y error, aprendiendo de la experiencia (slides1 p4).
- Las acciones no solo determinan la recompensa inmediata, también el siguiente estado y, con él, todas las recompensas futuras (slides1 p4).
- Se sitúa junto al aprendizaje supervisado (clasificación, regresión), el no supervisado y el semi-supervisado como un paradigma distinto (slides1 p5).
- Ejemplos citados: juegos (Atari/DQN, ajedrez y Go con AlphaGo/AlphaZero), robótica y control, modelos de lenguaje e IA generativa, y finanzas (gestión de carteras) (slides1 p6).

Lecturas de referencia: Sutton & Barto (2018), caps. 1–3; clases 1 y 2 de D. Silver (slides1 p3).

---

## 2. Elementos formales de un sistema de RL
| Elemento | Definición resumida | Fuente |
|---|---|---|
| Agente | Quien aprende y decide; maximiza la recompensa acumulada | slides1 p7 |
| Entorno | Todo lo externo al agente; responde a las acciones con una recompensa y un cambio de estado | slides1 p7 |
| Política $\pi$ | Mapeo de estados a acciones. Determinista: $a=\pi(s)$. Estocástica: $\pi(a\mid s)=P(A_t=a\mid S_t=s)$ | slides1 p8 |
| Recompensa $R_t\in\mathbb{R}$ | Escalar que da el entorno en cada paso; define el objetivo inmediato; puede ser estocástica | slides1 p9 |
| Función de valor | Recompensa total esperada a largo plazo desde un estado o par estado-acción. La recompensa es primaria e inmediata; el valor es secundario y a futuro, y es el que guía las acciones y por tanto la política | slides1 p10 |
| Modelo del entorno (opcional) | Representación que imita al entorno y sirve para planificar | slides1 p11 |

**Model-based frente a model-free** (slides1 p11): los algoritmos *model-based* usan un modelo para planificar (simular el futuro antes de actuar); los *model-free* aprenden directamente por ensayo y error sin construir un modelo explícito. Una taxonomía gráfica de algoritmos aparece en slides1 p12 (figura de OpenAI Spinning Up, no transcrita).

### Dilema exploración-explotación (slides1 p13)
- **Explotación:** elegir la acción con mayor recompensa esperada según el conocimiento actual.
- **Exploración:** elegir acciones subóptimas según la estimación actual para obtener información.
- Ninguna estrategia pura sirve: explotar desde el principio atrapa en óptimos locales; explorar siempre impide rentabilizar lo aprendido.

### Bandido multibrazo, la versión sin estados (slides1 p14–p21)
- $k$ acciones, cada una con recompensa esperada $q_*(a)=\mathbb{E}[R_{t+1}\mid A_t=a]$ desconocida. Las acciones no modifican el estado del entorno (slides1 p14).
- Estimación por promedio muestral: $Q_t(a)=\dfrac{\sum_{i=1}^{t-1}R_i\,\mathbb{I}(A_i=a)}{N_t(a)}$ (slides1 p15).
- **Greedy:** $A_t=\arg\max_a Q_t(a)$, empates al azar; nunca explora (slides1 p16).
- **$\epsilon$-greedy:** con probabilidad $1-\epsilon$ explota y con $\epsilon$ elige una acción al azar entre las $|A|$ (slides1 p17):
$$P(A_t=a)=\begin{cases}1-\epsilon+\frac{\epsilon}{|A|} & a=\arg\max_b Q_t(b)\\ \frac{\epsilon}{|A|} & \text{resto}\end{cases}$$
  A largo plazo puede convenir un $\epsilon\neq0$ (slides1 p17).
- **10-armed testbed:** $q_*(a)\sim\mathcal N(0,1)$ y $R\sim\mathcal N(q_*(a),1)$ (slides1 p18). Las gráficas de recompensa media y de % de acción óptima comparan $\epsilon=0$, $0{,}01$ y $0{,}1$ en 1000 pasos (slides1 p19–p20).
- **Actualización incremental** sin guardar el historial (slides1 p21):
$$Q_{n+1}=Q_n+\tfrac1n\,(R_n-Q_n)=Q_n\left(1-\tfrac1n\right)+\tfrac1n R_n$$

Nota: esta es exactamente la regla "estimación nueva = estimación vieja + paso × (objetivo − estimación vieja)" que reaparece en Monte Carlo, SARSA y Q-Learning.

---

## 3. El bucle agente-entorno y la propiedad de Markov
**Bucle de interacción** (slides1 p22): el agente (política $\pi$) envía la acción $A_t$; el entorno devuelve el estado $S_{t+1}$ y la recompensa $R_{t+1}$. Se genera una trayectoria
$$S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,\dots$$

**Propiedad de Markov** (slides1 p23–p24). El estado $S_t$ recoge toda la información relevante del pasado: la probabilidad del siguiente estado y recompensa depende solo de $(S_t,A_t)$, no de cómo se llegó a ellos:
$$P(S_{t+1}=s',R_{t+1}=r\mid S_t,A_t,S_{t-1},A_{t-1},\dots,S_0,A_0)=P(S_{t+1}=s',R_{t+1}=r\mid S_t,A_t)$$
- Entorno que la cumple = entorno **markoviano**; una tarea de RL en uno de ellos es un **MDP**; si estados y acciones son finitos, un **MDP finito (fMDP)** (slides1 p24).

**Dinámica del fMDP** (slides1 p25): una única distribución $p:\mathcal S\times\mathcal R\times\mathcal S\times\mathcal A\to[0,1]$,
$$p(s',r\mid s,a)=P(S_{t+1}=s',R_{t+1}=r\mid S_t=s,A_t=a).$$
De ella se derivan la probabilidad de transición $P(s'\mid s,a)=\sum_{r}p(s',r\mid s,a)$ y la recompensa esperada $R(s,a,s')=\sum_r r\,\frac{p(s',r\mid s,a)}{p(s'\mid s,a)}$.

---

## 4. Retorno y descuento (slides1 p26)
El agente maximiza el **retorno descontado**:
$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\dots=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad \gamma\in[0,1)$$
$\gamma$ es el factor de descuento y determina el valor presente de las recompensas futuras. En tareas episódicas con un número finito de transiciones, el ejemplo de clase usa $\gamma=1$ (slides1 p34). Para episodios que terminan en $T$, el retorno es $G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}$ (slides2 p31).

La condición de existencia y unicidad de la solución de las ecuaciones de Bellman es $\gamma<1$ o que toda trayectoria termine en un número finito de pasos (slides2 p11).

---

## 5. Funciones de valor
Bajo una política $\pi$ (slides1 p27–p29):
- **Valor de estado:** $V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$.
- **Valor de acción:** $Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$.
- Relaciones entre ambas:
$$V^\pi(s)=\sum_{a}\pi(a\mid s)\,Q^\pi(s,a),\qquad Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V^\pi(s')\,]$$

---

## 6. Ecuaciones de Bellman de expectativa (slides1 p30–p31)
Descomponen el valor en recompensa inmediata esperada más valor descontado del siguiente estado esperado:
$$V^\pi(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V^\pi(s')\,]$$
$$Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\Big[r+\gamma\sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a')\Big]$$
- La de $V^\pi$ es un **sistema de $|\mathcal S|$ ecuaciones lineales con $|\mathcal S|$ incógnitas**: se resuelve analíticamente (espacios pequeños) o con métodos iterativos de programación dinámica, y $V^\pi$ es su única solución (slides1 p30).

---

## 7. Política y funciones de valor óptimas
- **Orden parcial de políticas:** $\pi\ge\pi'$ si y solo si $V^\pi(s)\ge V^{\pi'}(s)\ \forall s$. Siempre existe al menos una política óptima $\pi_*$ mejor o igual que todas (slides1 p36).
- Puede haber varias óptimas, pero todas comparten $V_*(s)=\max_\pi V^\pi(s)$ y $Q_*(s,a)=\max_\pi Q^\pi(s,a)$, con $V_*(s)=\max_aQ_*(s,a)$ y $Q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma V_*(s')]$ (slides1 p37).

**Ecuaciones de optimalidad de Bellman** (slides1 p38): la media ponderada por $\pi$ se convierte en un máximo.
$$V_*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V_*(s')\,]$$
$$Q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)\Big[r+\gamma\max_{a'}Q_*(s',a')\Big]$$
- Es un sistema de $|\mathcal S|$ ecuaciones **no lineales** con solución única, que determina la política óptima: toda la probabilidad a la acción que maximiza recompensa más valor descontado del siguiente estado (slides1 p39).
- El $\max$ impide la solución directa por inversión matricial; hacen falta métodos iterativos (programación dinámica, Q-Learning, value iteration). **Los distintos métodos de RL difieren en cómo aproximan la solución de estas ecuaciones** (slides1 p39).

---

## 8. Ejemplo de clase: GridWorld de 4 estados (slides1 p32–p41)
**Definición** (slides1 p32–p34): estados $\{s_0,s_1,s_2,s_3\}$ con $s_2$ meta (terminal); acciones $\{D,A\}$ (derecha, abajo); episodios empiezan en $s_0$. Entorno determinista: desde $s_0$, $D\to s_1$ y $A\to s_2$; desde $s_1$, $D\to s_1$ (muro) y $A\to s_3$; desde $s_3$, $D\to s_3$ (muro) y $A\to s_2$. Recompensa $0$ al llegar a $s_2$ y $-1$ en el resto; $\gamma=1$.

**Evaluación de la política equiprobable** (slides1 p35):
$$V^\pi(s_0)=0{,}5V^\pi(s_1)+0{,}5V^\pi(s_2)-0{,}5$$
$$V^\pi(s_1)=0{,}5V^\pi(s_1)+0{,}5V^\pi(s_3)-1$$
$$V^\pi(s_2)=0,\qquad V^\pi(s_3)=0{,}5V^\pi(s_3)+0{,}5V^\pi(s_2)-0{,}5$$
Solución: $V^\pi=(-2,-3,0,-1)$.

**Optimalidad** (slides1 p40–p41): $V_*(s_0)=\max(-1+V_*(s_1),0)$, $V_*(s_1)=\max(-1+V_*(s_1),-1+V_*(s_3))$, $V_*(s_3)=\max(-1+V_*(s_3),0)$. Solución $V_*=(0,-1,0,0)$ con $\pi_*(s_0)=\pi_*(s_1)=\pi_*(s_3)=A$.

### Solución matricial e iterativa en clase (clase_20261023.ipynb)
La ecuación de expectativa se escribe como $V=AV+b$, con $A$ la matriz de transición bajo la política equiprobable ponderada por $\gamma=1$ y $b$ el vector de recompensas esperadas (celdas "A" y "b"):
$$A=\begin{pmatrix}0&.5&0&0\\0&.5&0&.5\\0&0&0&0\\0&0&0&.5\end{pmatrix},\qquad b=(-0{,}5,\,-1,\,0,\,-0{,}5)^\top$$
- **Directa:** $V=AV+b\Rightarrow(I-A)V=b$, resuelto con `np.linalg.solve(np.eye(4)-A, b)` (celdas "V = A V + b" y siguiente).
- **Iterativa:** se parte de `V = np.zeros((4,1))` y se repite `V = A @ V + b` durante **20 iteraciones** (celda "Solución iterativa").
- **Óptima:** de nuevo desde ceros, 20 iteraciones de las ecuaciones de optimalidad escritas a mano y actualizadas **in-place** (cada línea ya usa los valores actualizados): `V[0]=max(-1+V[1],0)`, `V[1]=max(-1+V[1],-1+V[3])`, `V[2]=0`, `V[3]=max(-1+V[3],0)` (celda "Política óptima").
- Sin criterio de parada: el número fijo de 20 iteraciones basta en este problema pequeño. Nota: la cota de la teoría para parar es $\max_s|V_{k+1}-V_k|<\theta$ (slides2 p13).

---

## 9. Ejercicio de GridWorld 5×5 (gridworld.ipynb, Sutton & Barto ej. 3.8)
**Entorno** (celda "Descripción del problema"): tablero $5\times5$; acciones $\{N,S,E,W\}$; una acción que sacaría del tablero deja al agente en su casilla con $R=-1$; en la casilla $A$ toda acción lleva a $A'$ con $R=10$; en $B$ lleva a $B'$ con $R=5$; el resto $R=0$; $\gamma=0{,}9$.

**Cuatro ejercicios** (el notebook trae el enunciado y celdas `# To do...`, sin solución):
1. Plantear la ecuación de expectativa de forma matricial para $\pi(a\mid s)=1/4$, con 25 estados numerados de 0 a 24 por filas. Ejemplo dado para $s_0$: $V^\pi(s_0)=\tfrac12(-1+\gamma V^\pi(s_0))+\tfrac14\gamma V^\pi(s_1)+\tfrac14\gamma V^\pi(s_5)$, es decir, $(1-\tfrac12\gamma)V(s_0)-\tfrac14\gamma V(s_1)-\tfrac14\gamma V(s_5)=-\tfrac12$.
2. Resolverla con `np.linalg.solve`.
3. Resolverla iterativamente desde `v = np.zeros((5,5))`.
4. Resolver iterativamente la ecuación de optimalidad desde ceros para obtener $V_*$ y la política óptima.

Es el mismo ejercicio propuesto en slides1 p42 (figura 3.1 de Sutton & Barto).

---

## 10. Ejemplos con solución óptima (slides1 p43–p48)
Se muestran, sin desarrollo teórico nuevo, los $V_*$ y $\pi_*$ de dos entornos de Gymnasium (se resuelven en `01_dp_y_montecarlo.md`):
- **FrozenLake-v1**, con `is_slippery=False` (valores en torno a 0,95–1,00 a lo largo del camino) y `is_slippery=True` (valores bastante menores, p. ej. 0,54 en el origen) (slides1 p44–p45). Nota: el efecto del deslizamiento es que la dinámica estocástica reduce el valor y cambia la política.
- **Taxi-v4**, mapas de $V_*$ y $\pi_*$ para pasajero en el taxi y para pasajero en la casilla azul (slides1 p47–p48).

---

## Aplicación al taller
Marco del taller: Q-Learning y SARSA para operar un activo, con estado discreto (bin de tendencia, posición) de 10 estados, acciones {vender, mantener, comprar}, recompensa = variación de precio según la posición menos una comisión del 0,1 %, y episodios de ~500 pasos con una serie nueva en cada episodio.

- **Correspondencia con el MDP.** Estado $s$ = (bin de tendencia, posición); acción $a$ = vender/mantener/comprar; recompensa $R_{t+1}$ = retorno de la posición menos comisión. El episodio es la trayectoria $S_0,A_0,R_1,\dots$ de ~500 pasos (slides1 p22). Se trata de una tarea episódica de horizonte finito.
- **Nota sobre la propiedad de Markov.** Con 10 estados discretizados es una aproximación: el bin de tendencia resume el pasado reciente, pero no garantiza Markov (slides1 p23–p24 definen la condición). Además, la posición sí afecta a la siguiente recompensa, lo que justifica incluirla en el estado.
- **Por qué no se resuelven las ecuaciones de Bellman directamente.** La solución lineal ($V=AV+b$) y la iteración sobre $V_*$ de clase_20261023 necesitan $p(s',r\mid s,a)$. En el taller la dinámica la genera una serie de precios nueva cada episodio y nadie conoce $p$. Por eso hace falta aprender de la experiencia (ver `01_dp_y_montecarlo.md` y `02_td_sarsa_qlearning.md`).
- **Qué se aproxima.** Q-Learning aproxima la ecuación de optimalidad de $Q_*$ (slides1 p38–p39); SARSA aproxima la de expectativa de $Q^\pi$ con $\pi$ $\epsilon$-greedy (slides1 p31). Ambos son los "diferentes métodos que difieren en cómo aproximan" la solución (slides1 p39).
- **Papel de $\gamma$.** Descuenta las recompensas futuras (slides1 p26). Con episodios finitos de ~500 pasos, $\gamma=1$ es válido (como en slides1 p34), pero $\gamma<1$ da prioridad a las ganancias próximas y reduce la varianza del retorno. Es un hiperparámetro a justificar en el informe (Nota: la elección concreta no viene de las fuentes).
- **Exploración.** Como en el bandido (slides1 p13, p17), una política puramente greedy se atasca; se usa $\epsilon$-greedy con $\epsilon$ decreciente o fijo. La actualización incremental $Q\leftarrow Q+\alpha(\text{objetivo}-Q)$ generaliza la de slides1 p21 con paso $\alpha$ en lugar de $1/n$.
