# Aprendizaje por diferencia temporal: TD(0), Sarsa y Q-Learning

> **Fuentes:** slides3 (`docs/raw/slides3.pdf`, "Métodos basados en Diferencia Temporal", L. F. Lago Fernández, MIAX-14), pp. 1–43. Lectura obligatoria indicada en la p3: Sutton & Barto (2018), cap. 6, secciones 6.1–6.6.
>
> **Convenciones:** "(slides3 pN)" = página N de las slides. "Nota" = contenido propio que no está en las slides. Las slides no cubren Expected Sarsa, Double Q-Learning ni n-step (ver §9).

---

## 1. Idea general de TD
**Idea.** TD combina lo mejor de DP y MC: como MC, no necesita modelo y aprende de experiencia muestreada; como DP, actualiza a partir de estimaciones previas (*bootstrapping*) sin esperar al final del episodio. Sirve para aprendizaje *online* y para tareas episódicas o continuas (slides3 p4).

Comparación de las tres familias tabulares (slides3 p5–p6):

| Propiedad | DP | MC | TD |
|---|---|---|---|
| Modelo $p$ | Sí | No | No |
| *Bootstrapping* | Sí | No | Sí |
| Cuándo actualiza | Por barrido completo | Al final del episodio | Paso a paso ($t+1$) |
| Entorno | Episódico y continuo | Solo episódico | Episódico y continuo |
| Varianza de la muestra | Nula (calcula la esperanza) | Alta (suma de $T-t$ recompensas) | Baja (un paso estocástico) |
| Sesgo inicial | Alto (depende de $V_0$) | Nulo | Alto (depende de $V_0$) |

*Fuente: slides3 p4–p6.*

---

## 2. Predicción TD(0)
**Idea.** El problema de predicción estima $V_\pi(s)$ para una política dada (slides3 p7). MC espera a $G_t$ y actualiza $V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]$ (p7). TD sustituye $G_t$ por una estimación: la recompensa observada más el valor descontado del siguiente estado (p9):

$$V(S_t)\leftarrow V(S_t)+\alpha\,[\,R_{t+1}+\gamma V(S_{t+1})-V(S_t)\,]$$

- **Objetivo TD** (*TD target*): $R_{t+1}+\gamma V(S_{t+1})$, que estima $G_t$ combinando una muestra real con la estimación actual (p10).
- **Error TD:** $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$. Es el error cometido en $t$ pero disponible en $t+1$ (p10).
- Con $\alpha=1/k$ en la $k$-ésima visita, MC calcula el promedio incremental de retornos; con $\alpha$ constante se obtiene *constant-$\alpha$ MC* (p8).

**Pseudocódigo TD(0)** (slides3 p16): inicializar $V(s)$ arbitrario (con $V(\text{terminal})=0$); por episodio, inicializar $S$; mientras $S$ no sea terminal: $A\leftarrow\pi(\cdot|S)$, tomar $A$, observar $R,S'$, aplicar la regla anterior y $S\leftarrow S'$.

*Fuente: slides3 p7–p10, p16.*

---

## 3. DP, MC y TD: qué aproxima cada uno
**Idea.** Todos parten de $V_\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]=\mathbb{E}_\pi[R_{t+1}+\gamma V_\pi(S_{t+1})|S_t=s]$ (ecuación de Bellman, slides3 p11).

- **DP:** esperanza exacta sobre acciones y sucesores con la dinámica conocida $p(s',r|s,a)$, pero usa $V(s')$ estimado (*bootstrapping*) (p12).
- **MC:** una trayectoria completa; retornos reales (sin *bootstrapping*) pero la esperanza se aproxima por el promedio de una muestra finita (p13).
- **TD(0):** un paso muestral más *bootstrapping*; acumula las dos aproximaciones (esperanza por muestra y $V_\pi(S_{t+1})$ por su estimación actual) (p14).

**Sesgo y varianza.** TD tiene menor varianza que MC porque depende de una sola transición $(R_{t+1},S_{t+1})$ y no de toda la secuencia (p17, p6). El sesgo inicial de TD depende de $V_0$ (p6); el de MC es nulo. *Nota:* las slides no desarrollan más el compromiso sesgo-varianza.

**Ventajas de TD(0)** (p17): sin modelo; totalmente *online* con actualización inmediata; aplicable a tareas continuas; menor varianza que MC.

*Fuente: slides3 p11–p15, p17.*

---

## 4. Convergencia de TD(0) y comparación con MC
**Condiciones.** TD(0) converge a $V_\pi$ cuando el número de pasos tiende a infinito si, para todo $s$ (slides3 p18):

$$\sum_{t=1}^{\infty}\alpha_t(s)=\infty\qquad\text{y}\qquad\sum_{t=1}^{\infty}\alpha_t^2(s)<\infty .$$

Con $\alpha$ constante, TD(0) no converge exactamente: oscila alrededor del valor verdadero con una amplitud que depende de $\alpha$ (p18).

**TD(0) vs constant-$\alpha$ MC.** Ambos convergen asintóticamente, pero TD(0) suele converger bastante más rápido con datos en lotes finitos (*batch training*); con $\alpha$ constante y pequeño convergen a soluciones distintas (p19).

**Ejemplo A-B** (p21–p26): 8 episodios, $\gamma=1$: (A,0,B,0), seis veces (B,1) y una (B,0). Ambos dan $V(B)=0{,}75$. MC da $V(A)=0$ (media de los retornos observados, que minimiza el error cuadrático medio sobre lo visto). TD(0) da $V(A)=0+V(B)=0{,}75$, porque construye implícitamente el **modelo de máxima verosimilitud** del MDP y obtiene su valor exacto en sentido DP. Como el problema es markoviano, la estimación de TD(0) predice mejor los retornos futuros con datos no vistos que la respuesta "memorizada" de MC (p25–p26).

*Fuente: slides3 p18–p26.*

---

## 5. Control TD: on-policy vs off-policy
**Idea.** Sin modelo, el control estima $Q(s,a)$ en lugar de $V(s)$. La regla general es (slides3 p27):

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,[\,\text{target}-Q(S_t,A_t)\,]$$

La naturaleza del *target* decide si el algoritmo es *on-policy* u *off-policy* (p27).

- **On-policy (Sarsa):** evalúa y optimiza la misma política que usa para actuar (p28).
- **Off-policy (Q-Learning):** optimiza una política distinta de la que usa para elegir acciones (p32).

*Fuente: slides3 p27, p28, p32.*

---

## 6. Sarsa (on-policy)
**Idea.** Usa la transición $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$, donde $A_{t+1}$ se elige con la política exploratoria actual (p. ej. $\epsilon$-greedy) (slides3 p28).

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,[\,R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\,]$$

$$\text{Target}_{\text{Sarsa}}=R_{t+1}+\gamma\,Q(S_{t+1},A_{t+1})\quad(\text{p29})$$

- Es *on-policy* porque la misma política decide el comportamiento (elegir $A_{t+1}$) y define el objetivo.
- La política debe ser $\epsilon$-greedy para garantizar la exploración, y el algoritmo converge a la mejor política entre las $\epsilon$-greedy (p30).

**Pseudocódigo** (slides3 p31): inicializar $Q$ arbitrario con $Q(\text{terminal},\cdot)=0$. Por episodio: inicializar $S$; elegir $A$ desde $S$ ($\epsilon$-greedy). Por paso: tomar $A$, observar $R,S'$; **elegir $A'$ desde $S'$** ($\epsilon$-greedy); actualizar $Q(S,A)$ con $R+\gamma Q(S',A')$; $S\leftarrow S'$, $A\leftarrow A'$. Parámetros requeridos: $\alpha\in(0,1]$, $\epsilon>0$, $\gamma\in[0,1]$.

*Fuente: slides3 p28–p31.*

---

## 7. Q-Learning (off-policy)
**Idea.** $Q$ aproxima directamente $Q^*$ y esto no depende de la política seguida para explorar, siempre que todos los pares estado-acción se visiten eventualmente (slides3 p32).

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\,[\,R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\,]$$

$$\text{Target}_{\text{Q-learning}}=R_{t+1}+\gamma\max_a Q(S_{t+1},a)\quad(\text{p33})$$

**Dos políticas** (p34):
- **Política objetivo:** puramente codiciosa respecto a la $Q$ actual, $a^*=\arg\max_a Q(S_{t+1},a)$.
- **Política de comportamiento:** normalmente estocástica y exploratoria, como $\epsilon$-greedy.

**Convergencia.** Converge a la política óptima si la política de comportamiento garantiza la visita de todos los pares estado-acción (p35).

**Pseudocódigo** (slides3 p36): inicializar $Q$ con $Q(\text{terminal},\cdot)=0$. Por episodio: inicializar $S$. Por paso: **elegir $A$ desde $S$** ($\epsilon$-greedy); tomar $A$, observar $R,S'$; actualizar con $R+\gamma\max_a Q(S',a)$; $S\leftarrow S'$. Aquí no se arrastra ningún $A'$: la acción se elige al inicio de cada paso.

*Fuente: slides3 p32–p36.*

---

## 8. Sarsa vs Q-Learning en *cliff walking*
**Entorno** (Sutton & Barto, ej. 6.6; slides3 p37–p39): cuadrícula con inicio S=(3,0), meta G=(3,11) y acantilado en las casillas (3,1)–(3,10). Acciones arriba/abajo/izquierda/derecha deterministas. Recompensa $-1$ por paso y $-100$ al entrar en el acantilado, que además devuelve al agente a S. Tarea episódica y sin descuento.

**Resultados.**
- Q-Learning encuentra la **ruta óptima** (junto al acantilado), porque la política que evalúa es determinista (p40).
- Sarsa encuentra una ruta **subóptima pero más segura** para una política $\epsilon$-greedy (p41).
- Rendimiento *online*: Q-Learning es peor porque eventualmente cae al acantilado al explorar; en la figura de la p42 la curva de recompensa de Sarsa queda por encima de la de Q-Learning durante los ~5000 episodios.
- Rendimiento asintótico: el de Q-Learning es óptimo en el límite; el de Sarsa es subóptimo si $\epsilon>0$ fijo y óptimo solo cuando $\epsilon\to 0$ (p43).

*Nota:* la diferencia viene del *target*: Sarsa incorpora en $Q(S',A')$ el coste esperado de los pasos exploratorios cerca del precipicio; Q-Learning supone que después actuará de forma codiciosa.

*Fuente: slides3 p37–p43.*

---

## 9. Hiperparámetros, inicialización y lo que las slides no cubren
- **$\alpha\in(0,1]$** (p16, p31, p36): tasa de aprendizaje. Convergencia exacta si $\sum\alpha_t=\infty$ y $\sum\alpha_t^2<\infty$; con $\alpha$ constante hay oscilación cuya amplitud depende de $\alpha$ (p18). Con $\alpha=1/k$ se obtiene el promedio incremental (p8).
- **$\epsilon>0$** (p31, p36): parámetro de exploración; $\epsilon$-greedy garantiza explorar todas las opciones (p30). Sarsa es óptimo solo si $\epsilon\to0$ (p43).
- **$\gamma\in[0,1]$** (p16, p31, p36): factor de descuento (cliff walking se plantea sin descuento, p39).
- **Inicialización de $Q$:** arbitraria para todo $s,a$, con $Q(\text{terminal},\cdot)=0$ (p31, p36); igual con $V$ en TD(0) (p16). El sesgo inicial de TD depende de $V_0$ (p6).
- **Lo que las slides NO dicen:** no mencionan GLIE ni calendarios concretos de decaimiento de $\epsilon$ o $\alpha$ (solo la condición de convergencia de $\alpha$ en p18 y "$\epsilon\to0$" en p43), ni Expected Sarsa, Double Q-Learning, n-step o TD($\lambda$). Cualquier uso de ellos sería *Nota* propia.

---

## Aplicación al taller

**Código del repo** (`src/agents.py`, `src/experiment.py`): `TabularAgent` ya aporta $Q$ (10 estados × 3 acciones, inicializada a 0), `act()` $\epsilon$-greedy con desempate aleatorio y `end_episode()` con $\epsilon\leftarrow\max(\epsilon_{\min},\epsilon\cdot\text{decay})$ por episodio. Solo falta implementar `update(s,a,r,s2,a2,done)`.

| Algoritmo | Clase | `a2` en `update` | Orden en `run_episode` | Slides |
|---|---|---|---|---|
| Sarsa | `SarsaAgent` (`on_policy=True`) | Acción ya elegida para `s2` | Elige $A'$, luego actualiza | p31 |
| Q-Learning | `QLearningAgent` | `None` | Actualiza, luego elige $A'$ con la $Q$ ya actualizada | p36 |

**Reglas exactas a implementar** ($Q(\text{terminal},\cdot)=0$ vía `done`, p31/p36):

- **Sarsa:** `target = r if done else r + gamma * Q[s2, a2]`; `Q[s, a] += alpha * (target - Q[s, a])`.
- **Q-Learning:** `target = r if done else r + gamma * Q[s2].max()`; `Q[s, a] += alpha * (target - Q[s, a])`.

El `max` de Q-Learning corresponde a la política objetivo codiciosa (p34); el desempate aleatorio de `act` solo afecta al comportamiento.

**Coherencia con las slides.** El orden de `run_episode` y el tratamiento de `done` coinciden con los pseudocódigos (p31, p36). Matices (*Nota*):
- Las slides piden $\epsilon>0$ en el pseudocódigo; el código lo decae por episodio hasta `epsilon_min=0.01`. Es compatible con "óptimo cuando $\epsilon\to0$" (p43), aunque con `epsilon_min>0` Sarsa queda ligeramente subóptimo en el límite.
- Cliff walking usa $\gamma=1$ y entorno determinista; aquí $\gamma<1$ (0,99 por defecto) y series nuevas cada episodio, así que con $\alpha$ constante habrá oscilación residual (p18).
- `evaluate(greedy=True)` ejecuta la política puramente codiciosa, que no es la $\epsilon$-greedy que Sarsa aprendió a evaluar (p30). Es el análogo de "ruta óptima vs ruta segura" y puede favorecer a Q-Learning en test.

**Comportamiento esperable en trading** (*Nota*, extrapolado de p40–p43):
- *Q-Learning:* converge a la política óptima de su $Q$ (p35, p43), pero mientras $\epsilon$ sea alto su recompensa de entrenamiento será peor que la de Sarsa (p42).
- *Sarsa:* sus valores incluyen el coste de las acciones exploratorias, así que cabe esperar políticas más conservadoras y mejor recompensa *online* con $\epsilon>0$ (p41–p42). La diferencia debería reducirse al decaer $\epsilon$.
- Con solo 10 estados la tabla es pequeña, y con comisión del 0,1 % y series ruidosas las diferencias son un resultado empírico a medir, no algo que las slides garanticen.

**Hiperparámetros a barrer** (los que las slides identifican como parámetros): $\alpha$ (p. ej. 0,01–0,5; menor $\alpha$ implica menos oscilación, p18), $\epsilon$ inicial, `epsilon_min` y `epsilon_decay` (cuándo $\epsilon\to0$, p43), $\gamma$ (p. ej. 0,9–0,99) y número de episodios. La inicialización de $Q$ queda en 0 por defecto (una inicialización optimista sería variante propia, no cubierta en las slides). Comparar ambos algoritmos con las mismas semillas en los 4 mercados (tendencia+ciclo, browniano, reversión a la media, regímenes).
